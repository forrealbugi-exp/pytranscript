"""Poprawianie tekstu – trzy backendy: Ollama, llama-cpp-python, LanguageTool."""
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# --- Ollama (opcjonalny) ---
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# --- LanguageTool (fallback) ---
try:
    import language_tool_python
    HAS_LT = True
except ImportError:
    HAS_LT = False

# --- llama-cpp-python (fallback bez serwera) ---
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

# --- huggingface_hub (do pobierania modeli) ---
try:
    from huggingface_hub import hf_hub_download
    HAS_HF = True
except ImportError:
    HAS_HF = False


# Domyślny mały model GGUF (szybki, działa na CPU)
DEFAULT_GGUF_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
DEFAULT_GGUF_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


# ============================================================
# AUTO-START OLLAMA
# ============================================================

_ollama_process = None
_ollama_lock = threading.Lock()


def is_ollama_running(host="127.0.0.1", port=11434, timeout=0.8) -> bool:
    """Sprawdza, czy serwer Ollama nasłuchuje."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def ensure_ollama_running(wait_seconds: float = 15.0) -> bool:
    """
    Sprawdza, czy Ollama działa. Jeśli nie – próbuje uruchomić 'ollama serve'
    w tle i czeka, aż serwer będzie gotowy.
    Zwraca True, jeśli Ollama działa.
    """
    global _ollama_process

    if is_ollama_running():
        return True

    if not HAS_OLLAMA:
        return False

    # Znajdź ścieżkę do ollama.exe
    ollama_exe = _find_ollama_executable()
    if ollama_exe is None:
        print("[ollama] Nie znaleziono pliku ollama.exe")
        return False

    with _ollama_lock:
        if _ollama_process is not None and _ollama_process.poll() is None:
            # Już próbowaliśmy uruchomić – czekamy
            pass
        else:
            try:
                # Flagi do ukrycia okna na Windowsie
                creationflags = 0
                if sys.platform == "win32":
                    creationflags = subprocess.CREATE_NO_WINDOW

                print(f"[ollama] Uruchamiam: {ollama_exe} serve")
                _ollama_process = subprocess.Popen(
                    [str(ollama_exe), "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
            except Exception as e:
                print(f"[ollama] Nie udało się uruchomić: {e}")
                return False

    # Czekaj aż port 11434 zacznie odpowiadać
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if is_ollama_running():
            print("[ollama] Serwer gotowy.")
            return True
        time.sleep(0.5)

    print("[ollama] Serwer nie odpowiedział w czasie.")
    return False


def _find_ollama_executable() -> Path | None:
    """Szuka ollama.exe / ollama w typowych lokalizacjach."""
    import shutil
    # 1. W PATH
    p = shutil.which("ollama")
    if p:
        return Path(p)

    # 2. Windows – domyślna lokalizacja instalatora
    if sys.platform == "win32":
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path("C:/Program Files/Ollama/ollama.exe"),
            Path("C:/Program Files (x86)/Ollama/ollama.exe"),
        ]
        for c in candidates:
            if c.exists():
                return c

    # 3. macOS
    if sys.platform == "darwin":
        for c in [Path("/usr/local/bin/ollama"),
                  Path("/opt/homebrew/bin/ollama")]:
            if c.exists():
                return c

    return None


# ============================================================
# POSTPROCESSOR
# ============================================================

PROMPT_TEMPLATE = (
    "You are a professional transcription editor. "
    "Fix grammar, spelling, punctuation, and sentence structure. "
    "Merge fragmented sentences into coherent ones when appropriate "
    "(e.g. 'This is a day. I've been looking forward to. two and a half years.' "
    "should become 'This is the day I've been looking forward to for two and a half years.'). "
    "Preserve the original meaning and language. "
    "Do not add information that isn't implied. "
    "Do not add any explanations, notes, or quotes. "
    "Return ONLY the corrected text, nothing else.\n\n"
    "TEXT TO CORRECT:\n{text}"
)


class TextPostProcessor:
    """Korekcja tekstu – Ollama (jeśli działa) → llama-cpp → LanguageTool."""

    def __init__(self, model: str = "gemma3", use_ollama: bool = True,
                 gguf_repo: str = DEFAULT_GGUF_REPO,
                 gguf_file: str = DEFAULT_GGUF_FILE):
        self.model = model
        self.use_ollama = use_ollama and HAS_OLLAMA
        self.gguf_repo = gguf_repo
        self.gguf_file = gguf_file

        self._model_checked = False
        self._llama = None            # instancja llama-cpp
        self._llama_lock = threading.Lock()

    # ---------- Status / wykrywanie backendu ----------

    def active_backend(self) -> str:
        """Zwraca 'ollama', 'llama-cpp', 'languagetool' lub 'none'."""
        if self.use_ollama and is_ollama_running():
            return "ollama"
        if HAS_LLAMA_CPP and HAS_HF:
            return "llama-cpp"
        if HAS_LT:
            return "languagetool"
        return "none"

    def ollama_status(self) -> tuple[bool, str]:
        """(ok, message) – do wyświetlenia w Ustawieniach."""
        if not HAS_OLLAMA:
            return False, "Brak pakietu 'ollama' w Pythonie. pip install ollama"
        if not is_ollama_running():
            # Spróbuj uruchomić
            if ensure_ollama_running():
                pass
            else:
                return False, ("Serwer Ollama nie działa i nie udało się go "
                               "uruchomić. Uruchom ręcznie: ollama serve")
        models = self.list_ollama_models()
        if not models:
            return False, ("Ollama działa, ale nie masz modelu. "
                           "Uruchom: ollama pull gemma3")
        return True, models[0]

    def list_ollama_models(self) -> list[str]:
        if not self.use_ollama or not is_ollama_running():
            return []
        try:
            resp = ollama.list()
            raw = resp.get("models", [])
            return [m.get("model") or m.get("name")
                    for m in raw if (m.get("model") or m.get("name"))]
        except Exception as e:
            print(f"[postprocess] Błąd listy modeli: {e}")
            return []

    def _ensure_model(self):
        if self._model_checked:
            return
        self._model_checked = True
        models = self.list_ollama_models()
        if not models:
            return
        wanted = self.model
        if wanted in models:
            return
        for m in models:
            if m.startswith(wanted + ":"):
                self.model = m
                return
        self.model = models[0]

    # ---------- Główna metoda ----------

    def correct(self, text: str, language: str = "auto") -> str:
        if not text.strip():
            return text

        # 1. Ollama
        if self.use_ollama:
            if not is_ollama_running():
                ensure_ollama_running()
            if is_ollama_running():
                self._ensure_model()
                try:
                    return self._correct_ollama(text)
                except Exception as e:
                    print(f"[postprocess] Ollama error: {e}")

        # 2. llama-cpp-python
        if HAS_LLAMA_CPP and HAS_HF:
            try:
                return self._correct_llama_cpp(text)
            except Exception as e:
                print(f"[postprocess] llama-cpp error: {e}")

        # 3. LanguageTool
        if HAS_LT:
            try:
                return self._correct_languagetool(text, language)
            except Exception as e:
                print(f"[postprocess] LanguageTool error: {e}")

        return text

    # ---------- Ollama ----------

    def _correct_ollama(self, text: str) -> str:
        prompt = PROMPT_TEMPLATE.format(text=text)
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.15, "num_predict": 2048},
        )
        result = response["message"]["content"].strip()
        return self._clean(result)

    # ---------- llama-cpp ----------

    def _get_llama(self) -> "Llama | None":
        """Ładuje model GGUF (pobiera raz z Hugging Face)."""
        with self._llama_lock:
            if self._llama is not None:
                return self._llama

            # Pobierz model do cache Hugging Face
            try:
                print(f"[llama-cpp] Pobieram {self.gguf_repo}/{self.gguf_file} …")
                model_path = hf_hub_download(
                    repo_id=self.gguf_repo,
                    filename=self.gguf_file,
                )
            except Exception as e:
                print(f"[llama-cpp] Nie udało się pobrać modelu: {e}")
                return None

            print(f"[llama-cpp] Ładuję model z {model_path} …")
            self._llama = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=max(2, (os.cpu_count() or 4) - 1),
                verbose=False,
            )
            print("[llama-cpp] Model gotowy.")
            return self._llama

    def _correct_llama_cpp(self, text: str) -> str:
        llm = self._get_llama()
        if llm is None:
            raise RuntimeError("Nie udało się załadować modelu llama-cpp.")

        prompt = PROMPT_TEMPLATE.format(text=text)
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            max_tokens=1024,
        )
        result = response["choices"][0]["message"]["content"].strip()
        return self._clean(result)

    # ---------- LanguageTool ----------

    def _correct_languagetool(self, text: str, language: str) -> str:
        lang_map = {"pl": "pl-PL", "uk": "uk-UA", "en": "en-US"}
        lang_code = lang_map.get(language, "en-US")
        tool = language_tool_python.LanguageTool(lang_code)
        matches = tool.check(text)
        return language_tool_python.utils.correct(text, matches)

    # ---------- Pomocnicze ----------

    @staticmethod
    def _clean(result: str) -> str:
        """Usuwa ogrodzenia markdown i nadmiarowe cudzysłowy."""
        if result.startswith("```"):
            lines = [l for l in result.split("\n")
                     if not l.strip().startswith("```")]
            result = "\n".join(lines).strip()
        # Usuń opakowanie w cudzysłowy, jeśli model dodał
        if len(result) > 1 and result[0] == result[-1] and result[0] in "\"'":
            result = result[1:-1].strip()
        return result