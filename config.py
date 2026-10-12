"""Zarządzanie ustawieniami aplikacji."""
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from storage import get_app_dir

SETTINGS_FILE = get_app_dir() / "settings.json"


@dataclass
class Settings:
    source: str = "mic"
    language: str = "auto"
    model: str = "base"
    ignore_crowd: bool = False
    ignore_background_noise: bool = False
    auto_correct: bool = False
    live_ai_correct: bool = False
    postprocess_model: str = "gemma3"
    app_language: str = "pl"
    live_silence_ms: int = 700

    @classmethod
    def load(cls) -> "Settings":
        try:
            if not SETTINGS_FILE.exists():
                return cls()
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            allowed = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**allowed)
        except Exception as e:
            print(f"[config] Błąd wczytywania: {e}")
            return cls()

    def save(self) -> None:
        try:
            SETTINGS_FILE.write_text(
                json.dumps(asdict(self), indent=2, ensure_ascii=False),
                encoding="utf-8")
        except Exception as e:
            print(f"[config] Błąd zapisu: {e}")

    @classmethod
    def reset(cls) -> "Settings":
        s = cls()
        s.save()
        return s