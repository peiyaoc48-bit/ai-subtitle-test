"""被测服务：字幕翻译 API。

启动：uvicorn app.main:app --reload
文档：http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.srt import SrtFormatError, parse_srt, to_srt
from app.translator import get_translator

SUPPORTED_LANGS = {"en", "ja", "ko", "th", "es"}
MAX_CUES = 500

app = FastAPI(title="AI 字幕翻译服务")


class TranslateRequest(BaseModel):
    srt: str
    target_lang: str = "en"


class TranslateResponse(BaseModel):
    srt: str
    cue_count: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    if req.target_lang not in SUPPORTED_LANGS:
        raise HTTPException(400, f"不支持的目标语言：{req.target_lang}")
    try:
        cues = parse_srt(req.srt)
    except SrtFormatError as e:
        raise HTTPException(400, str(e))
    if len(cues) > MAX_CUES:
        raise HTTPException(413, f"字幕条数超过上限 {MAX_CUES}")

    translated = get_translator().translate(cues, req.target_lang)
    return TranslateResponse(srt=to_srt(translated), cue_count=len(translated))
