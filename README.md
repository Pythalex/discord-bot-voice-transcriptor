# voice-transcribe

A Discord bot that transcribes voice messages to text, fully locally on the GPU using
faster-whisper (Whisper large-v3). Forward a voice message into the bot's channel and it
replies with the transcript. No audio leaves your machine.

This project was made using Claude.

## Setup

### 1. Create the bot (Discord Developer Portal)

1. Go to <https://discord.com/developers/applications> → **New Application**, give it a name.
2. Left sidebar → **Bot**.
   - Click **Reset Token** → **Copy** the token (this is your `DISCORD_TOKEN`; treat it like a password).
   - Scroll to **Privileged Gateway Intents** → enable **MESSAGE CONTENT INTENT** → Save.
3. Left sidebar → **OAuth2** → **URL Generator**:
   - Scopes: **bot**
   - Bot Permissions: **View Channels**, **Send Messages**, **Read Message History**, **Attach Files**
   - Open the generated URL in your browser and add the bot to your server.

### 2. Get the channel ID

In Discord: **User Settings → Advanced → Developer Mode** (on). Then right-click your bot
channel → **Copy Channel ID**.

### 3. Configure

```bash
cp .env.example .env
```
Edit `.env` and set `DISCORD_TOKEN` and `CHANNEL_ID`.

### 4. Run

```bash
./run-bot.sh
```
Wait for `logged in as ... — watching channel <id>`.

## Usage

- **Forward** any voice message (hover/right-click → Forward → pick the bot channel). The bot
  replies with the transcript.
- Or upload an audio file (`.ogg`, `.mp3`, `.m4a`, `.wav`, ...) directly into the channel.

The reply has a header (`📝 lang · duration · time`) followed by the text. Transcripts too long
for one Discord message (>2000 chars) come back as an attached `.txt`.

## Configuration (`.env`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `DISCORD_TOKEN` | — | Bot token (required). |
| `CHANNEL_ID` | — | Only react in this channel. Empty = every channel the bot can see. |
| `WHISPER_MODEL` | `large-v3` | Model: `tiny`/`base`/`small`/`medium`/`large-v3`. |
| `LANGUAGE` | `fr` | Forced language code, or `auto` to detect. |

## Notes

- The Whisper model (~2.9 GB) is cached in `~/.cache/huggingface/` and is not re-downloaded between runs.
- GPU works via the pip-installed CUDA libs (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`);
  `run-bot.sh` puts them on the loader path. Falls back to CPU automatically if no compatible GPU.
