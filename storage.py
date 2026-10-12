"""Zarządzanie plikami sesji transkrypcji w folderze aplikacji."""
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path


APP_NAME = "pytranscript"


def get_app_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    d = base / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_sessions_dir() -> Path:
    d = get_app_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_srt_dir() -> Path:
    d = get_app_dir() / "srt"
    d.mkdir(parents=True, exist_ok=True)
    return d


def text_to_srt(text: str) -> str:
    sentences = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in ".!?":
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    lines = []
    for i, sent in enumerate(sentences, start=1):
        s = (i - 1) * 3
        e = i * 3
        lines += [str(i), f"00:00:{s:02d},000 --> 00:00:{e:02d},000", sent, ""]
    return "\n".join(lines)


class Session:
    def __init__(self, id=None, title="", text="", created=None, updated=None,
                 language="auto", source="mic", model="base"):
        self.id = id or uuid.uuid4().hex[:12]
        self.title = title
        self.text = text
        self.created = created or datetime.now().isoformat(timespec="seconds")
        self.updated = updated or self.created
        self.language = language
        self.source = source
        self.model = model

    def to_dict(self):
        return {
            "id": self.id, "title": self.title, "text": self.text,
            "created": self.created, "updated": self.updated,
            "language": self.language, "source": self.source, "model": self.model,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d.get("id"),
            title=d.get("title", ""),
            text=d.get("text", ""),
            created=d.get("created"),
            updated=d.get("updated"),
            language=d.get("language", "auto"),
            source=d.get("source", "mic"),
            model=d.get("model", "base"),
        )

    def file_path(self) -> Path:
        return get_sessions_dir() / f"{self.id}.json"

    def srt_path(self) -> Path:
        return get_srt_dir() / f"{self.id}.srt"

    def save(self):
        self.updated = datetime.now().isoformat(timespec="seconds")
        try:
            self.file_path().write_text(
                json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[storage] Zapisz JSON: {e}")
        try:
            self.srt_path().write_text(text_to_srt(self.text), encoding="utf-8")
        except Exception as e:
            print(f"[storage] Zapisz SRT: {e}")

    def delete(self):
        for p in (self.file_path(), self.srt_path()):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    def display_title(self, max_len=26) -> str:
        t = (self.title or "").strip().replace("\n", " ")
        if not t:
            # fallback: pierwszy fragment tekstu
            t = (self.text or "").strip().replace("\n", " ")
            if not t:
                return "(bez tytułu)"
        if len(t) > max_len:
            t = t[:max_len].rsplit(" ", 1)[0] + "…"
        return t

    def display_date(self) -> str:
        try:
            dt = datetime.fromisoformat(self.updated)
            return dt.strftime("%m-%d %H:%M")
        except Exception:
            return self.updated or ""


def load_sessions() -> list[Session]:
    sessions = []
    for f in get_sessions_dir().glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            sessions.append(Session.from_dict(d))
        except Exception as e:
            print(f"[storage] Nie udało się wczytać {f.name}: {e}")
    sessions.sort(key=lambda s: s.updated, reverse=True)
    return sessions