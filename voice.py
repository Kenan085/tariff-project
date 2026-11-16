"""
Voice layer for accessibility (blind / low-vision users, motor impairments, low literacy).

Speech-to-text:  1) OpenAI Whisper if OPENAI_API_KEY is set (best for Azerbaijani)
                 2) Google Web Speech via SpeechRecognition (free, no key, needs internet)
Text-to-speech:  1) edge-tts neural voices (free, no key; has real Azerbaijani voices)
                 2) browser speechSynthesis (works offline, voice depends on the OS)
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re

STT_LANG = {"az": "az-AZ", "en": "en-US", "ru": "ru-RU"}
TTS_VOICE = {"az": "az-AZ-BanuNeural", "en": "en-US-JennyNeural", "ru": "ru-RU-SvetlanaNeural"}


def speech_to_text(wav_bytes: bytes, lang: str) -> str | None:
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            f = io.BytesIO(wav_bytes); f.name = "speech.wav"
            return OpenAI().audio.transcriptions.create(model="whisper-1", file=f, language=lang).text
        except Exception as exc:
            print("Whisper error:", exc)
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(wav_bytes)) as src:
            audio = r.record(src)
        return r.recognize_google(audio, language=STT_LANG[lang])
    except Exception as exc:
        print("STT error:", exc)
        return None


def clean_for_speech(text: str, lang: str) -> str:
    """Remove emojis / markdown, and say currency the way a person would."""
    text = re.sub(r"[*_#`>|]", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)
    if lang == "az":
        text = text.replace("AZN", "manat").replace("GB", "giqabayt")
    elif lang == "ru":
        text = text.replace("AZN", "манат").replace("GB", "гигабайт")
    else:
        text = text.replace("AZN", "manat")
    return text.strip()


def text_to_speech(text: str, lang: str, rate: str = "+0%") -> bytes | None:
    """MP3 bytes from edge-tts, or None (caller then uses browser speech)."""
    try:
        import edge_tts

        async def _run() -> bytes:
            buf = bytearray()
            comm = edge_tts.Communicate(clean_for_speech(text, lang), TTS_VOICE[lang], rate=rate)
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf.extend(chunk["data"])
            return bytes(buf)

        data = asyncio.run(_run())
        return data or None
    except Exception as exc:
        print("TTS error:", exc)
        return None


def browser_tts_html(text: str, lang: str, rate: float = 1.0) -> str:
    """Fallback: speak with the browser's built-in voices (no network)."""
    return f"""<script>
    const u = new SpeechSynthesisUtterance({json.dumps(clean_for_speech(text, lang))});
    u.lang = {json.dumps(STT_LANG[lang])}; u.rate = {rate};
    window.parent.speechSynthesis.cancel(); window.parent.speechSynthesis.speak(u);
    </script>"""
