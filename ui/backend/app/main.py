import json
import re
from io import BytesIO
from dotenv import load_dotenv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import db
from .rag_service import RagService
from .settings import get_settings


class ChatCreate(BaseModel):
    title: Optional[str] = None


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4_000)
    show_assets: Optional[bool] = None
    use_lcel: Optional[bool] = None


class ChatResponse(BaseModel):
    id: str
    title: Optional[str]
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    chat_id: str
    role: str
    content: str
    created_at: str
    sources: Optional[List[Dict[str, Any]]] = None
    attachments: Optional[List[Dict[str, Any]]] = None


class ChatDetailResponse(BaseModel):
    chat: ChatResponse
    messages: List[MessageResponse]


# Load .env once at startup so backend can read env vars automatically.
load_dotenv()
settings = get_settings()
db.init_db(settings.db_path)
rag_service = RagService(settings)
uploads_dir = settings.root_dir / "data" / "processed" / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GridVision Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_TABLE_STYLE_MARKER = "data-gv-table-style"
_TABLE_STYLE = """
:root {
  color-scheme: light;
}
body {
  margin: 0;
  padding: 12px;
  background: #f5f7fb;
  color: #0f172a;
  font-family: "Source Serif 4", "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino,
    "Times New Roman", serif;
}
.table-wrap {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
  overflow: auto;
}
table {
  width: 100%;
  min-width: 480px;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 13px;
  line-height: 1.45;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
}
.table-wrap table {
  border: 0;
  border-radius: 0;
}
th,
td {
  padding: 8px 10px;
  border-bottom: 1px solid #e2e8f0;
  border-right: 1px solid #eef2f7;
  text-align: left;
  vertical-align: top;
  white-space: pre-wrap;
  word-break: break-word;
}
th:last-child,
td:last-child {
  border-right: 0;
}
tbody tr:nth-child(even) td {
  background: #f8fafc;
}
th {
  background: #eef2f7;
  color: #334155;
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 0.05em;
}
td {
  font-variant-numeric: tabular-nums;
}
@media (max-width: 640px) {
  body {
    padding: 8px;
  }
  table {
    min-width: 360px;
    font-size: 12px;
  }
}
""".strip()


def _wrap_table_html(html: str) -> str:
    raw = html.strip()
    if not raw:
        return html
    if _TABLE_STYLE_MARKER in raw:
        return raw
    style_tag = f"<style {_TABLE_STYLE_MARKER}=\"1\">{_TABLE_STYLE}</style>"
    lower = raw.lower()
    if "<html" in lower:
        head_close = lower.find("</head>")
        if head_close != -1:
            return raw[:head_close] + style_tag + raw[head_close:]
        body_open = lower.find("<body")
        if body_open != -1:
            body_end = lower.find(">", body_open)
            if body_end != -1:
                return raw[: body_end + 1] + style_tag + raw[body_end + 1 :]
        return style_tag + raw
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        f"  {style_tag}\n"
        "</head>\n"
        "<body>\n"
        "  <div class=\"table-wrap\">\n"
        f"{raw}\n"
        "  </div>\n"
        "</body>\n"
        "</html>\n"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _trim_title(content: str, max_len: int = 60) -> str:
    title = content.strip().replace("\n", " ")
    if len(title) <= max_len:
        return title
    return title[: max_len - 3] + "..."


_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_UPLOAD_NAME_RE = re.compile(r"^[a-f0-9]{32}\.(?:jpg|png|webp)$")
_IMAGE_FORMAT_SUFFIXES = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_MAX_UPLOAD_PIXELS = 40_000_000


def _is_english_query(text: str) -> bool:
    if _CYRILLIC_RE.search(text):
        return False
    return True


def _validate_query(text: str, *, allow_empty: bool = False) -> str:
    content = text.strip()
    if not content and not allow_empty:
        raise HTTPException(status_code=400, detail="Message content is empty")
    if len(content) > 4_000:
        raise HTTPException(status_code=413, detail="Message content exceeds 4,000 characters")
    if content and not _is_english_query(content):
        raise HTTPException(status_code=400, detail="Only English queries are allowed.")
    return content


async def _read_valid_image_upload(file: UploadFile) -> tuple[bytes, str]:
    image_bytes = await file.read(_MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")
    if len(image_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MiB")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            if image.width * image.height > _MAX_UPLOAD_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions are too large")
            suffix = _IMAGE_FORMAT_SUFFIXES.get(image.format or "")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Unsupported or invalid image")

    if suffix is None:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, and WebP images are allowed")
    return image_bytes, suffix


def _message_to_response(message: db.MessageRecord) -> MessageResponse:
    return MessageResponse(
        id=message.id,
        chat_id=message.chat_id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        sources=message.sources,
        attachments=message.attachments,
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "time": _now_iso()}

@app.get("/status")
def status() -> Dict[str, Any]:
    return {"models": rag_service.model_status(), "time": _now_iso()}


@app.get("/chats", response_model=List[ChatResponse])
def list_chats() -> List[ChatResponse]:
    chats = db.list_chats(settings.db_path)
    return [ChatResponse(**db.as_dict(chat)) for chat in chats]


@app.post("/chats", response_model=ChatResponse)
def create_chat(payload: ChatCreate) -> ChatResponse:
    chat = db.create_chat(settings.db_path, title=payload.title)
    return ChatResponse(**db.as_dict(chat))


@app.get("/chats/{chat_id}", response_model=ChatDetailResponse)
def get_chat(chat_id: str) -> ChatDetailResponse:
    chat = db.get_chat(settings.db_path, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    messages = db.list_messages(settings.db_path, chat_id)
    return ChatDetailResponse(
        chat=ChatResponse(**db.as_dict(chat)),
        messages=[_message_to_response(m) for m in messages],
    )


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str) -> Dict[str, str]:
    chat = db.get_chat(settings.db_path, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.delete_chat(settings.db_path, chat_id)
    return {"status": "ok"}


@app.post("/chats/{chat_id}/messages")
def post_message(chat_id: str, payload: MessageCreate) -> Dict[str, Any]:
    chat = db.get_chat(settings.db_path, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    content = _validate_query(payload.content)

    title = _trim_title(content)
    db.update_chat_title_if_empty(settings.db_path, chat_id, title)

    user_msg = db.add_message(settings.db_path, chat_id, "user", content)

    history = [db.as_dict(m) for m in db.list_messages(settings.db_path, chat_id)]
    answer = rag_service.answer(
        content,
        history=history,
        force_assets=payload.show_assets,
        use_lcel=payload.use_lcel,
    )

    assistant_msg = db.add_message(
        settings.db_path,
        chat_id,
        "assistant",
        answer.answer,
        sources=answer.sources,
        attachments=answer.attachments,
    )

    return {
        "user_message": _message_to_response(user_msg),
        "assistant_message": _message_to_response(assistant_msg),
        "debug": answer.debug,
    }


@app.post("/chats/{chat_id}/messages/stream")
async def stream_message(chat_id: str, payload: MessageCreate, request: Request) -> StreamingResponse:
    chat = db.get_chat(settings.db_path, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    content = _validate_query(payload.content)

    title = _trim_title(content)
    db.update_chat_title_if_empty(settings.db_path, chat_id, title)

    user_msg = db.add_message(settings.db_path, chat_id, "user", content)

    history = [db.as_dict(m) for m in db.list_messages(settings.db_path, chat_id)]
    answer = rag_service.answer(
        content,
        history=history,
        force_assets=payload.show_assets,
        use_lcel=payload.use_lcel,
    )

    assistant_msg = db.add_message(
        settings.db_path,
        chat_id,
        "assistant",
        answer.answer,
        sources=answer.sources,
        attachments=answer.attachments,
    )

    async def event_stream():
        start = {
            "type": "start",
            "assistant_message_id": assistant_msg.id,
            "user_message": _message_to_response(user_msg).model_dump(),
        }
        yield f"data: {json.dumps(start)}\n\n"

        chunk_size = 60
        text = assistant_msg.content
        for i in range(0, len(text), chunk_size):
            if await request.is_disconnected():
                break
            token = text[i : i + chunk_size]
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        final_payload = {
            "type": "final",
            "assistant_message": _message_to_response(assistant_msg).model_dump(),
            "debug": answer.debug,
        }
        yield f"data: {json.dumps(final_payload)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/chats/{chat_id}/messages/image")
async def post_image_message(
    chat_id: str,
    file: UploadFile = File(...),
    content: str = Form(""),
    show_assets: Optional[bool] = Form(None),
) -> Dict[str, Any]:
    chat = db.get_chat(settings.db_path, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    text = _validate_query(content, allow_empty=True)

    title = _trim_title(text or "Image prompt")
    db.update_chat_title_if_empty(settings.db_path, chat_id, title)

    image_bytes, suffix = await _read_valid_image_upload(file)
    upload_id = uuid4().hex
    upload_name = f"{upload_id}{suffix}"
    upload_path = uploads_dir / upload_name
    upload_path.write_bytes(image_bytes)

    user_attachment = {
        "asset_id": f"upload_{upload_id}",
        "type": "user_upload",
        "file_path": str(upload_path.relative_to(settings.root_dir)),
        "page_label": "upload",
        "element_id": "",
        "url": f"/uploads/{upload_name}",
        "title": "User upload",
    }

    user_msg = db.add_message(
        settings.db_path,
        chat_id,
        "user",
        text or "[Image prompt]",
        attachments=[user_attachment],
    )
    history = [db.as_dict(m) for m in db.list_messages(settings.db_path, chat_id)]
    answer = rag_service.answer_with_image(
        image_bytes,
        query=text,
        history=history,
        force_assets=show_assets,
    )

    assistant_msg = db.add_message(
        settings.db_path,
        chat_id,
        "assistant",
        answer.answer,
        sources=answer.sources,
        attachments=answer.attachments,
    )

    return {
        "user_message": _message_to_response(user_msg),
        "assistant_message": _message_to_response(assistant_msg),
        "debug": answer.debug,
    }


@app.get("/assets/{asset_id}")
def get_asset(asset_id: str):
    path = rag_service.resolve_asset_path(asset_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    if path.suffix.lower() == ".html":
        html = path.read_text(encoding="utf-8", errors="replace")
        return HTMLResponse(
            _wrap_table_html(html),
            headers={
                "Content-Security-Policy": (
                    "sandbox; default-src 'none'; style-src 'unsafe-inline'; "
                    "img-src 'self' data:"
                )
            },
        )
    return FileResponse(path)


@app.get("/uploads/{filename}")
def get_upload(filename: str):
    if not _UPLOAD_NAME_RE.fullmatch(filename):
        raise HTTPException(status_code=404, detail="Upload not found")
    path = uploads_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Upload not found")
    return FileResponse(path)
