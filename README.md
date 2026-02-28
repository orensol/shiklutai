# Telegram Video Transcription App (Farsi/Arabic -> English)

This MVP accepts a Telegram video URL and returns:
- detected language
- original transcript
- English translation
- English summary

## Requirements

- Python 3.11+
- `ffmpeg` installed and available in `PATH`
- OpenAI API key

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000

Sample URL to test:

`https://t.me/eretzh/27602`

## API

`POST /process`

Body:

```json
{
  "url": "https://t.me/eretzh/27602"
}
```

Response fields:
- `source_url`
- `video_url` (playable preview URL served by the app)
- `detected_language`
- `transcript_original`
- `translation_en`
- `summary_en`

## Notes

- The app uses `yt-dlp` to download from Telegram public links.
- If a Telegram link is private or blocked, download will fail.
- Language detection is optimized for Arabic (`ar`) and Farsi (`fa`) but will return other languages when detected.
