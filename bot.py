#!/usr/bin/env python3
"""Discord bot that transcribes audio attachments to text.

Drop (or forward) a voice message / audio file into the watched channel and the
bot replies with the transcript. Uses the same warm faster-whisper model as the
POC, kept in VRAM for the bot's lifetime.

Config comes from a .env file (see .env.example):
    DISCORD_TOKEN   - bot token (required, keep secret)
    CHANNEL_ID      - only react in this channel id (optional; empty = every
                      channel the bot can see)
    WHISPER_MODEL   - model name (default: large-v3)
    LANGUAGE        - forced language code, e.g. fr (default: fr; "auto" to detect)
"""
import asyncio
import functools
import io
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import discord
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
CHANNEL_ID = os.getenv("CHANNEL_ID", "").strip()
CHANNEL_ID = int(CHANNEL_ID) if CHANNEL_ID.isdigit() else None
MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3").strip()
LANGUAGE = os.getenv("LANGUAGE", "fr").strip()
LANGUAGE = None if LANGUAGE.lower() == "auto" else LANGUAGE

DISCORD_LIMIT = 2000
AUDIO_EXTS = {".ogg", ".mp3", ".wav", ".m4a", ".flac", ".webm", ".oga", ".opus"}

# Transcription is GPU-bound and blocking; run it in a single worker thread so it
# never blocks the asyncio event loop, and so only one clip transcribes at a time
# (keeps VRAM use predictable).
_executor = ThreadPoolExecutor(max_workers=1)


def pick_backend():
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception as e:
        print(f"[warn] CUDA unavailable ({e}); using CPU", file=sys.stderr)
    return "cpu", "int8"


print(f"[info] loading model '{MODEL_NAME}'...", file=sys.stderr)
_device, _compute = pick_backend()
_t0 = time.time()
model = WhisperModel(MODEL_NAME, device=_device, compute_type=_compute)
print(f"[info] model ready on {_device} ({_compute}) in {time.time() - _t0:.1f}s", file=sys.stderr)


def _transcribe_sync(path: str) -> tuple[str, str, float]:
    """Blocking transcription. Returns (text, detected_language, duration_seconds)."""
    segments, info = model.transcribe(path, language=LANGUAGE, vad_filter=True, beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.language, info.duration


def is_audio(att: discord.Attachment) -> bool:
    if att.content_type and att.content_type.startswith("audio"):
        return True
    return Path(att.filename).suffix.lower() in AUDIO_EXTS


intents = discord.Intents.default()
intents.message_content = True  # required to see attachments/content; enable in the portal too
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    where = f"channel {CHANNEL_ID}" if CHANNEL_ID else "every visible channel"
    print(f"[info] logged in as {client.user} — watching {where}", file=sys.stderr)


def collect_audio_attachments(message: discord.Message) -> list[discord.Attachment]:
    """Audio attachments from the message itself AND from any forwarded message.

    A forwarded message (Discord's "Forward" feature) carries its attachments in
    message snapshots rather than in `message.attachments`, so we look in both.
    """
    atts = list(message.attachments)
    for snap in getattr(message, "message_snapshots", []) or []:
        atts.extend(getattr(snap, "attachments", []) or [])
    return [a for a in atts if is_audio(a)]


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return
    if CHANNEL_ID and message.channel.id != CHANNEL_ID:
        return

    audio_atts = collect_audio_attachments(message)
    if not audio_atts:
        return

    for att in audio_atts:
        try:
            await handle_attachment(message, att)
        except Exception as e:
            print(f"[error] {att.filename}: {e}", file=sys.stderr)
            await message.reply(f"⚠️ Transcription failed for `{att.filename}`: {e}",
                                mention_author=False)


async def handle_attachment(message: discord.Message, att: discord.Attachment):
    # Download to a temp file so PyAV can seek/decode it.
    suffix = Path(att.filename).suffix or ".ogg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        await att.save(tmp.name)

    try:
        async with message.channel.typing():
            loop = asyncio.get_running_loop()
            t0 = time.time()
            text, lang, duration = await loop.run_in_executor(
                _executor, functools.partial(_transcribe_sync, tmp_path)
            )
            elapsed = time.time() - t0
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not text:
        await message.reply("🔇 (no speech detected)", mention_author=False)
        return

    header = f"📝 **Transcription** · `{lang}` · {duration:.0f}s audio · {elapsed:.1f}s\n"
    body = header + text
    if len(body) <= DISCORD_LIMIT:
        await message.reply(body, mention_author=False)
    else:
        # Too long for one message: post the header + attach the full text as a file.
        file = discord.File(io.BytesIO(text.encode("utf-8")),
                            filename=f"{Path(att.filename).stem}.txt")
        await message.reply(header + "_(full transcript attached)_",
                            file=file, mention_author=False)


if __name__ == "__main__":
    if not TOKEN:
        sys.exit("DISCORD_TOKEN is not set. Copy .env.example to .env and fill it in.")
    client.run(TOKEN)
