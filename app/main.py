import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langdetect import detect
from openai import OpenAI
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv


load_dotenv(override=True)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Telegram Video Transcribe/Translate/Summarize")
VIDEO_CACHE_DIR = Path("static/video_cache")
VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class ProcessRequest(BaseModel):
    url: HttpUrl


class ProcessResponse(BaseModel):
    source_url: HttpUrl
    video_url: str
    detected_language: str
    transcript_original: str
    translation_en: str
    summary_en: str


def _safe_check_call(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{exc.stderr}") from exc


def download_video(url: str, output_dir: Path) -> Path:
    output_template = str(output_dir / "video.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-o",
        output_template,
        url,
    ]
    _safe_check_call(cmd)

    candidates = sorted(output_dir.glob("video.*"))
    if not candidates:
        raise RuntimeError("Video download succeeded but no output file was found.")
    return candidates[0]


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    audio_path = output_dir / "audio.mp3"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        str(audio_path),
    ]
    _safe_check_call(cmd)
    if not audio_path.exists():
        raise RuntimeError("Audio extraction failed; audio file was not created.")
    return audio_path


def transcribe_media(media_path: Path) -> str:
    with media_path.open("rb") as media_file:
        transcript = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=media_file,
        )

    text = getattr(transcript, "text", "") or ""
    if not text.strip():
        raise RuntimeError("Transcription returned empty text.")
    return text


def detect_language_hint(transcript: str) -> str:
    # Quick script heuristic first for Arabic/Persian script ranges.
    arabic_script_count = len(re.findall(r"[\u0600-\u06FF]", transcript))
    if arabic_script_count > max(20, len(transcript) // 10):
        try:
            lang = detect(transcript)
            if lang in {"fa", "ar"}:
                return lang
        except Exception:
            pass

    try:
        lang = detect(transcript)
        if lang in {"fa", "ar", "en"}:
            return lang
        return f"{lang} (non-target)"
    except Exception:
        return "unknown"


def translate_to_english(transcript: str, detected_language: str) -> str:
    prompt = (
        "Translate the following transcript to fluent, faithful English. "
        f"Detected source language: {detected_language}. "
        "Do not summarize; preserve meaning and details."
    )

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise translator."},
            {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript}"},
        ],
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def summarize_english(english_text: str) -> str:
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You summarize transcripts. Make no assumptions and add no external details.",
            },
            {
                "role": "user",
                "content": f"Summarize the English transcription. Make no assumptions.\n\nTranscript:\n{english_text}",
            },
        ],
    )
    content = response.choices[0].message.content or ""
    return content.strip()


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.post("/process", response_model=ProcessResponse)
def process_video(req: ProcessRequest) -> ProcessResponse:
    if shutil.which("yt-dlp") is None:
        raise HTTPException(status_code=500, detail="yt-dlp is not installed or not in PATH.")
    has_ffmpeg = shutil.which("ffmpeg") is not None

    with tempfile.TemporaryDirectory(prefix="tg-video-") as tmp:
        tmp_dir = Path(tmp)
        try:
            video_path = download_video(str(req.url), tmp_dir)
            cached_video_name = f"{uuid4().hex}{video_path.suffix or '.mp4'}"
            cached_video_path = VIDEO_CACHE_DIR / cached_video_name
            shutil.copy2(video_path, cached_video_path)
            video_url = f"/static/video_cache/{cached_video_name}"
            if has_ffmpeg:
                media_path = extract_audio(video_path, tmp_dir)
            else:
                media_path = video_path
            transcript = transcribe_media(media_path)
            detected_language = detect_language_hint(transcript)
            translation_en = translate_to_english(transcript, detected_language)
            summary_en = summarize_english(translation_en)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ProcessResponse(
        source_url=req.url,
        video_url=video_url,
        detected_language=detected_language,
        transcript_original=transcript,
        translation_en=translation_en,
        summary_en=summary_en,
    )
