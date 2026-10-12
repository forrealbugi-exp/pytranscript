"""pytranscript – chat UI z sesjami, nagrywaniem, transkrypcją LIVE i AI."""
import ctypes
import os
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def get_dpi_scale() -> float:
    if sys.platform != "win32":
        return 1.0
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return dpi / 96.0
    except Exception:
        return 1.0


from config import Settings
from audio import AudioRecorder, LiveAudioRecorder
from transcription import Transcriber
from languages import LANGUAGES, flag
from postprocess import (
    TextPostProcessor, is_ollama_running, ensure_ollama_running,
    HAS_OLLAMA, HAS_LLAMA_CPP, HAS_HF, HAS_LT,
)
from i18n import translate, APP_LANGUAGES
from storage import (
    Session, load_sessions, get_app_dir, get_srt_dir, text_to_srt,
)


SCALE = get_dpi_scale()


def px(value: int) -> int:
    return max(1, int(round(value * SCALE)))


def open_in_file_manager(path: Path):
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        messagebox.showerror("Błąd", str(e))


class LiveSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.final_text = ""
        self.pending = ""

    def add_raw(self, text: str) -> None:
        with self.lock:
            self.pending = (self.pending + " " + text).strip()

    def has_pending(self) -> bool:
        with self.lock:
            return bool(self.pending.strip())

    def snapshot_for_correction(self) -> str:
        with self.lock:
            return (self.final_text + " " + self.pending).strip()

    def apply_correction(self, corrected: str):
        with self.lock:
            self.final_text = corrected
            self.pending = ""
            return self.final_text, self.pending

    def current(self):
        with self.lock:
            return self.final_text, self.pending


class App:
    def __init__(self, root):
        self.root = root
        self.settings = Settings.load()

        # Model
        self.transcriber = None
        self.model_ready = False
        self.is_loading_model = False

        # Recording state
        self.recorder = None
        self.live_recorder = None
        self.live_session: LiveSession | None = None
        self.is_recording = False
        self.is_transcribing = False
        self.is_live = False
        self.live_stop_event = None

        # AI
        self.is_correcting = False
        self.postprocessor = TextPostProcessor(
            model=self.settings.postprocess_model, use_ollama=True)

        # Sessions
        self.sessions: list[Session] = load_sessions()
        self.current_session: Session | None = None
        self._loading_session = False
        self._save_after_id = None

        # Auto-start Ollama w tle
        threading.Thread(target=self._autostart_ollama, daemon=True).start()

        self._setup_window()
        self._build_ui()
        self._bind_shortcuts()

        if self.sessions:
            self._load_session_by_id(self.sessions[0].id)
        else:
            self.new_session()
        self._refresh_sessions_list(
            keep_id=self.current_session.id if self.current_session else None)

        # Ładuj model Whisper
        self.root.after(200, self._load_model_async)

    def t(self, key: str, **kwargs) -> str:
        return translate(key, self.settings.app_language, **kwargs)

    # ---------- Auto-start Ollama ----------

    def _autostart_ollama(self):
        """Uruchamia serwer Ollama w tle, jeśli nie działa."""
        try:
            ensure_ollama_running(wait_seconds=20.0)
        except Exception as e:
            print(f"[autostart] Ollama: {e}")

    # ---------- Okno ----------

    def _setup_window(self):
        self.root.title(self.t("app_title"))
        self.root.geometry(f"{px(1100)}x{px(720)}")
        self.root.minsize(px(820), px(540))
        try:
            self.root.tk.call("tk", "scaling", SCALE * 1.2)
        except Exception:
            pass
        style = ttk.Style()
        try:
            style.theme_use("vista" if sys.platform == "win32" else "clam")
        except tk.TclError:
            pass
        default_font = ("Segoe UI" if sys.platform == "win32" else "Helvetica", 11)
        self.root.option_add("*Font", default_font)

    def _build_ui(self):
        root = self.root
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # ============ SIDEBAR ============
        sidebar = ttk.Frame(root, padding=(10, 12, 6, 12))
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.configure(width=px(260))
        sidebar.grid_propagate(False)

        ttk.Button(sidebar, text="➕  " + self.t("new_session"),
                   command=self.new_session).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(sidebar, text=self.t("sessions_header"),
                  font=("", 9, "bold"), foreground="#888"
                  ).pack(anchor="w", padx=4, pady=(0, 4))

        list_wrap = ttk.Frame(sidebar)
        list_wrap.pack(fill=tk.BOTH, expand=True)

        self.sessions_listbox = tk.Listbox(
            list_wrap, activestyle="none", font=("", 10),
            selectmode=tk.SINGLE, borderwidth=0, highlightthickness=1,
            highlightbackground="#DDDDDD", highlightcolor="#007AFF",
        )
        self.sessions_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(list_wrap, command=self.sessions_listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.sessions_listbox.config(yscrollcommand=sb.set)

        self.sessions_listbox.bind("<<ListboxSelect>>", self._on_session_selected)
        self.sessions_listbox.bind("<Button-3>", self._on_session_right_click)
        if sys.platform == "darwin":
            self.sessions_listbox.bind("<Button-2>", self._on_session_right_click)
            self.sessions_listbox.bind("<Control-Button-1>",
                                       self._on_session_right_click)

        self.session_menu = tk.Menu(self.root, tearoff=0)
        self.session_menu.add_command(label=self.t("rename_session"),
                                      command=self.rename_session)
        self.session_menu.add_command(label=self.t("delete_session"),
                                      command=self.delete_session)
        self.session_menu.add_separator()
        self.session_menu.add_command(label=self.t("export_srt"),
                                      command=self.export_session_srt)

        ttk.Button(sidebar, text=self.t("open_folder"),
                   command=lambda: open_in_file_manager(get_app_dir())
                   ).pack(fill=tk.X, pady=(10, 0))

        # ============ MAIN PANEL ============
        main = ttk.Frame(root)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        header = ttk.Frame(main, padding=(12, 10, 12, 6))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        self.title_var = tk.StringVar()
        self.title_entry = ttk.Entry(header, textvariable=self.title_var,
                                     font=("", 14, "bold"))
        self.title_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.title_entry.bind("<FocusOut>", self._on_title_changed)
        self.title_entry.bind("<Return>", self._on_title_changed)

        self.source_lbl = ttk.Label(header, text="", foreground="#666",
                                    font=("", 10))
        self.source_lbl.grid(row=0, column=1, sticky="e", padx=8)

        ttk.Button(header, text=self.t("settings_btn"),
                   command=self.open_settings).grid(row=0, column=2, sticky="e")

        self.text_area = ScrolledText(main, wrap=tk.WORD, undo=True,
                                      font=("", 12), padx=10, pady=10,
                                      borderwidth=0, highlightthickness=0)
        self.text_area.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        self.text_area.tag_configure("pending", foreground="#999999")
        self.text_area.bind("<<Modified>>", self._on_text_modified)

        btns = ttk.Frame(main, padding=(12, 4, 12, 6))
        btns.grid(row=2, column=0, sticky="ew")
        btns.columnconfigure(8, weight=1)

        self.record_btn = ttk.Button(btns, text=self.t("record"),
                                     command=self.toggle_recording,
                                     state=tk.DISABLED)
        self.record_btn.grid(row=0, column=0, padx=(0, 6))

        self.live_btn = ttk.Button(btns, text=self.t("live"),
                                   command=self.toggle_live,
                                   state=tk.DISABLED)
        self.live_btn.grid(row=0, column=1, padx=(0, 6))

        ttk.Button(btns, text=self.t("copy"),
                   command=self.copy_text).grid(row=0, column=2, padx=3)
        ttk.Button(btns, text=self.t("clear"),
                   command=self.clear_text).grid(row=0, column=3, padx=3)
        self.correct_btn = ttk.Button(btns, text=self.t("ai_fix"),
                                      command=self.apply_ai_correction)
        self.correct_btn.grid(row=0, column=4, padx=3)

        status = ttk.Frame(main, padding=(12, 0, 12, 10))
        status.grid(row=3, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value=self.t("ready"))
        ttk.Label(status, textvariable=self.status_var,
                  foreground="#555", font=("", 10)
                  ).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(status, mode="indeterminate",
                                        length=px(140))
        self.progress.grid(row=0, column=1, sticky="e", padx=(8, 0))

    def _bind_shortcuts(self):
        self.root.bind("<Control-r>", lambda e: self.toggle_recording())
        self.root.bind("<Control-l>", lambda e: self.toggle_live())
        self.root.bind("<Control-s>", lambda e: self._force_save())
        self.root.bind("<Control-Shift-C>", lambda e: self.copy_text())
        self.root.bind("<Control-g>", lambda e: self.apply_ai_correction())
        self.root.bind("<Control-n>", lambda e: self.new_session())

    # ============================================================
    # SESSIONS
    # ============================================================

    def _refresh_sessions_list(self, keep_id: str | None = None):
        self.sessions_listbox.delete(0, tk.END)
        self._session_ids = []
        for s in self.sessions:
            label = f"{s.display_title()}   ·   {s.display_date()}"
            self.sessions_listbox.insert(tk.END, label)
            self._session_ids.append(s.id)
        if keep_id:
            for i, sid in enumerate(self._session_ids):
                if sid == keep_id:
                    self.sessions_listbox.selection_clear(0, tk.END)
                    self.sessions_listbox.selection_set(i)
                    self.sessions_listbox.see(i)
                    break

    def _on_session_selected(self, _event=None):
        if self._loading_session:
            return
        sel = self.sessions_listbox.curselection()
        if not sel:
            return
        sid = self._session_ids[sel[0]]
        if self.current_session and self.current_session.id == sid:
            return
        self._load_session_by_id(sid)

    def _on_session_right_click(self, event):
        idx = self.sessions_listbox.nearest(event.y)
        if idx < 0:
            return
        self.sessions_listbox.selection_clear(0, tk.END)
        self.sessions_listbox.selection_set(idx)
        try:
            self.session_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.session_menu.grab_release()

    def _load_session_by_id(self, sid: str):
        self._loading_session = True
        try:
            self._save_current_session()

            s = next((x for x in self.sessions if x.id == sid), None)
            if s is None:
                return
            self.current_session = s
            self.title_var.set(s.title or "")
            self.text_area.delete("1.0", tk.END)
            if s.text:
                self.text_area.insert("1.0", s.text)
            self.text_area.edit_modified(False)
            self._refresh_sessions_list(keep_id=s.id)
            self._set_status(self.t("ready"))
        finally:
            self._loading_session = False

    def new_session(self):
        self._save_current_session()
        s = Session(
            language=self.settings.language,
            source=self.settings.source,
            model=self.settings.model,
        )
        s.save()
        self.sessions.insert(0, s)
        self._load_session_by_id(s.id)
        self._refresh_sessions_list(keep_id=s.id)
        self.title_entry.focus_set()

    def _on_title_changed(self, _event=None):
        if self.current_session is None:
            return
        new_title = self.title_var.get().strip()
        if new_title != self.current_session.title:
            self.current_session.title = new_title
            self.current_session.save()
            self._refresh_sessions_list(keep_id=self.current_session.id)

    def _save_current_session(self):
        if self.current_session is None:
            return
        text = self.text_area.get("1.0", tk.END).rstrip()
        self.current_session.text = text
        if not self.current_session.title and text.strip():
            t = text.strip().replace("\n", " ")
            if len(t) > 40:
                t = t[:40].rsplit(" ", 1)[0] + "…"
            self.current_session.title = t
            self.title_var.set(t)
        self.current_session.language = self.settings.language
        self.current_session.source = self.settings.source
        self.current_session.model = self.settings.model
        self.current_session.save()

    def _on_text_modified(self, _event=None):
        if not self.text_area.edit_modified():
            return
        self.text_area.edit_modified(False)
        if self._loading_session:
            return
        if self._save_after_id is not None:
            try:
                self.root.after_cancel(self._save_after_id)
            except Exception:
                pass
        self._save_after_id = self.root.after(800, self._auto_save)

    def _auto_save(self):
        self._save_after_id = None
        if self.current_session is None:
            return
        self._save_current_session()
        try:
            idx = self._session_ids.index(self.current_session.id)
            label = (f"{self.current_session.display_title()}   ·   "
                     f"{self.current_session.display_date()}")
            self.sessions_listbox.delete(idx)
            self.sessions_listbox.insert(idx, label)
            self.sessions_listbox.selection_clear(0, tk.END)
            self.sessions_listbox.selection_set(idx)
        except Exception:
            pass

    def _force_save(self):
        self._save_current_session()
        self._set_status(self.t("saved_auto"))

    def rename_session(self):
        if self.current_session is None:
            return
        self.title_entry.focus_set()
        self.title_entry.select_range(0, tk.END)

    def delete_session(self):
        if self.current_session is None:
            return
        if not messagebox.askyesno(self.t("info"), self.t("confirm_delete")):
            return
        sid = self.current_session.id
        self.current_session.delete()
        self.sessions = [s for s in self.sessions if s.id != sid]
        self.current_session = None
        self.text_area.delete("1.0", tk.END)
        self.title_var.set("")
        if self.sessions:
            self._load_session_by_id(self.sessions[0].id)
        else:
            self.new_session()
        self._refresh_sessions_list(
            keep_id=self.current_session.id if self.current_session else None)
        self._set_status(self.t("session_deleted"))

    def export_session_srt(self):
        if self.current_session is None:
            return
        self._save_current_session()
        path = filedialog.asksaveasfilename(
            defaultextension=".srt",
            filetypes=[("SRT", "*.srt")],
            initialfile=f"{self.current_session.display_title()}.srt",
        )
        if not path:
            return
        try:
            Path(path).write_text(text_to_srt(self.current_session.text),
                                  encoding="utf-8")
            self._set_status(self.t("srt_saved", name=Path(path).name))
        except Exception as e:
            messagebox.showerror(self.t("srt_error"), str(e))

    # ============================================================
    # MODEL
    # ============================================================

    def _set_status(self, text: str):
        self.status_var.set(text)

    def _start_progress(self):
        self.progress.start(12)

    def _stop_progress(self):
        self.progress.stop()

    def _load_model_async(self):
        if self.is_loading_model or self.model_ready:
            return
        self.is_loading_model = True
        model = self.settings.model
        self._set_status(self.t("loading_model", model=model))
        self._start_progress()
        self.record_btn.config(state=tk.DISABLED)
        self.live_btn.config(state=tk.DISABLED)

        def worker():
            import traceback
            try:
                t = Transcriber(model)
                ok = t.load()
                err = t.load_error if not ok else None
            except Exception as e:
                t = None
                ok = False
                err = f"{e}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._on_model_loaded(t, ok, err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_model_loaded(self, transcriber, ok, err):
        self.is_loading_model = False
        self._stop_progress()
        if ok:
            self.transcriber = transcriber
            self.model_ready = True
            self.record_btn.config(state=tk.NORMAL)
            self.live_btn.config(state=tk.NORMAL)
            self._set_status(self.t("model_ready", model=transcriber.model_name))
        else:
            self._set_status(self.t("model_error"))
            messagebox.showerror(self.t("model_error"), err or "?")

    # ============================================================
    # RECORDING
    # ============================================================

    def toggle_recording(self):
        if self.is_transcribing or self.is_loading_model or self.is_live:
            return
        if self.is_recording:
            self._stop_and_transcribe()
        else:
            self._start_recording()

    def _start_recording(self):
        if not self.model_ready:
            messagebox.showinfo(self.t("info"), self.t("model_still_loading"))
            return
        self.recorder = AudioRecorder(source=self.settings.source)
        try:
            self.recorder.start()
        except Exception as e:
            messagebox.showerror(self.t("error"),
                                 self.t("start_recording_error", error=e))
            return
        self.is_recording = True
        self.record_btn.config(text=self.t("stop"))
        self._set_status(self.t("recording"))

    def _stop_and_transcribe(self):
        wav = self.recorder.stop() if self.recorder else None
        errors = self.recorder.errors if self.recorder else []
        self.is_recording = False
        self.record_btn.config(text=self.t("record"))

        if errors:
            messagebox.showwarning(self.t("info"), "\n".join(errors))
        if not wav:
            self._set_status(self.t("no_text"))
            return

        self.is_transcribing = True
        self.record_btn.config(state=tk.DISABLED)
        self.live_btn.config(state=tk.DISABLED)
        self._set_status(self.t("transcribing"))
        self._start_progress()
        threading.Thread(target=self._transcribe_worker,
                         args=(wav,), daemon=True).start()

    def _transcribe_worker(self, wav_path):
        try:
            text = self.transcriber.transcribe(
                wav_path, self.settings.language,
                ignore_crowd=self.settings.ignore_crowd,
                ignore_noise=self.settings.ignore_background_noise)
            self.root.after(0, self._on_transcription_done, text)
        except Exception as e:
            self.root.after(0, self._on_transcription_error, str(e))

    def _on_transcription_done(self, text: str):
        self._stop_progress()
        self.is_transcribing = False
        self.record_btn.config(state=tk.NORMAL)
        self.live_btn.config(state=tk.NORMAL)
        if not text:
            self._set_status(self.t("no_speech"))
            return
        self._set_status(self.t("ready"))
        self._append_text(text)
        self._force_save()
        if self.settings.auto_correct:
            self._apply_ai_correction_async(
                self.text_area.get("1.0", tk.END).strip())

    def _on_transcription_error(self, msg: str):
        self._stop_progress()
        self.is_transcribing = False
        self.record_btn.config(state=tk.NORMAL)
        self.live_btn.config(state=tk.NORMAL)
        self._set_status(self.t("transcription_error"))
        messagebox.showerror(self.t("transcription_error"), msg)

    # ============================================================
    # AI CORRECTION
    # ============================================================

    def apply_ai_correction(self):
        text = self.text_area.get("1.0", tk.END).strip()
        if not text:
            self._set_status(self.t("no_text"))
            return
        self._apply_ai_correction_async(text)

    def _apply_ai_correction_async(self, text: str):
        if self.is_correcting:
            return
        # Sprawdź dostępne backendy
        if not is_ollama_running() and not (HAS_LLAMA_CPP and HAS_HF) and not HAS_LT:
            self._set_status(self.t("ai_not_available"))
            return
        self.is_correcting = True
        self.correct_btn.config(state=tk.DISABLED)
        self._set_status(self.t("correcting"))
        self._start_progress()

        def worker():
            try:
                corrected = self.postprocessor.correct(text, self.settings.language)
                self.root.after(0, self._on_correction_done, corrected)
            except Exception as e:
                self.root.after(0, self._on_correction_error, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_correction_done(self, corrected: str):
        self._stop_progress()
        self.is_correcting = False
        self.correct_btn.config(state=tk.NORMAL)
        current = self.text_area.get("1.0", tk.END).strip()
        if corrected and corrected != current:
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert("1.0", corrected)
            self._force_save()
            self._set_status(self.t("corrected"))
        else:
            self._set_status(self.t("no_changes_ai"))

    def _on_correction_error(self, msg: str):
        self._stop_progress()
        self.is_correcting = False
        self.correct_btn.config(state=tk.NORMAL)
        self._set_status(self.t("correction_error"))
        messagebox.showerror(self.t("correction_error"), msg)

    # ============================================================
    # LIVE
    # ============================================================

    def toggle_live(self):
        if self.is_transcribing or self.is_loading_model or self.is_recording:
            return
        if self.is_live:
            self._stop_live()
        else:
            self._start_live()

    def _start_live(self):
        if not self.model_ready:
            messagebox.showinfo(self.t("info"), self.t("model_still_loading"))
            return

        self.live_recorder = LiveAudioRecorder(source=self.settings.source)
        try:
            self.live_recorder.start()
        except Exception as e:
            messagebox.showerror(self.t("error"),
                                 self.t("start_live_error", error=e))
            return

        self.live_session = LiveSession()
        self.is_live = True
        self.live_btn.config(text=self.t("stop"))
        self.record_btn.config(state=tk.DISABLED)
        self._set_status(self.t("listening"))

        self.live_stop_event = threading.Event()

        threading.Thread(
            target=self.transcriber.transcribe_stream,
            args=(self.live_recorder.queue, self.settings.language,
                  self._on_live_raw, self._on_live_error, self.live_stop_event),
            kwargs={
                "ignore_crowd": self.settings.ignore_crowd,
                "ignore_noise": self.settings.ignore_background_noise,
                "silence_ms": self.settings.live_silence_ms,
            },
            daemon=True,
        ).start()

        if self.settings.live_ai_correct and is_ollama_running():
            threading.Thread(
                target=self._live_ai_worker,
                args=(self.live_session, self.live_stop_event),
                daemon=True,
            ).start()

    def _stop_live(self):
        self.is_live = False
        if self.live_recorder:
            self.live_recorder.stop()
            if self.live_recorder.errors:
                messagebox.showwarning(self.t("info"),
                                       "\n".join(self.live_recorder.errors))
        if self.live_stop_event:
            self.live_stop_event.set()

        if self.live_session is not None:
            final, pending = self.live_session.current()
            merged = (final + " " + pending).strip()
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert("1.0", merged)
            self._force_save()

        if (self.settings.live_ai_correct and self.live_session
                and self.live_session.has_pending()
                and (is_ollama_running() or (HAS_LLAMA_CPP and HAS_HF))):
            threading.Thread(target=self._final_live_correction,
                             daemon=True).start()

        self.live_btn.config(text=self.t("live"))
        self.record_btn.config(state=tk.NORMAL)
        self._set_status(self.t("stopped"))

    def _final_live_correction(self):
        try:
            text = self.live_session.snapshot_for_correction()
            if not text:
                return
            corrected = self.postprocessor.correct(text, self.settings.language)
            self.live_session.apply_correction(corrected)
            self.root.after(0, self._refresh_live_display)
            self.root.after(0, self._force_save)
        except Exception as e:
            print(f"[live-ai-final] {e}")

    def _on_live_raw(self, text: str):
        if self.settings.live_ai_correct and self.live_session is not None:
            self.live_session.add_raw(text)
            self.root.after(0, self._refresh_live_display)
        else:
            self.root.after(0, lambda: self._append_text(text))
        self.root.after(0, self._force_save)

    def _on_live_error(self, msg: str):
        self.root.after(0, lambda: self._set_status(f"{self.t('error')}: {msg}"))

    def _live_ai_worker(self, session: LiveSession, stop_event: threading.Event):
        while not stop_event.is_set():
            stop_event.wait(4.0)
            if stop_event.is_set():
                break
            if not session.has_pending():
                continue
            try:
                text = session.snapshot_for_correction()
                if not text:
                    continue
                corrected = self.postprocessor.correct(text, self.settings.language)
                session.apply_correction(corrected)
                self.root.after(0, self._refresh_live_display)
            except Exception as e:
                print(f"[live-ai] {e}")

    def _refresh_live_display(self):
        if self.live_session is None:
            return
        final, pending = self.live_session.current()
        self._loading_session = True
        try:
            self.text_area.delete("1.0", tk.END)
            if final:
                self.text_area.insert(tk.END, final)
            if pending:
                if final:
                    self.text_area.insert(tk.END, " ")
                start = self.text_area.index(tk.END)
                self.text_area.insert(tk.END, pending)
                end = self.text_area.index(tk.END)
                self.text_area.tag_add("pending", start, end)
            self.text_area.see(tk.END)
            self.text_area.edit_modified(False)
        finally:
            self._loading_session = False

    # ============================================================
    # TEXT
    # ============================================================

    def _append_text(self, text: str):
        current = self.text_area.get("1.0", tk.END).strip()
        if current:
            self.text_area.insert(tk.END, " " + text)
        else:
            self.text_area.insert(tk.END, text)
        self.text_area.see(tk.END)

    def copy_text(self):
        text = self.text_area.get("1.0", tk.END).strip()
        if not text:
            self._set_status(self.t("no_text"))
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status(self.t("copied"))

    def clear_text(self):
        self.text_area.delete("1.0", tk.END)
        self._force_save()
        self._set_status(self.t("cleared"))

    # ============================================================
    # SETTINGS
    # ============================================================

    def open_settings(self):
        if self.is_live or self.is_recording:
            messagebox.showinfo(self.t("info"), self.t("stop_first"))
            return
        SettingsWindow(self.root, self.settings,
                       on_save=self._apply_settings, t_func=self.t)

    def _apply_settings(self):
        self.source_lbl.config(text=self._source_label())
        self.transcriber = None
        self.model_ready = False
        self.postprocessor = TextPostProcessor(
            model=self.settings.postprocess_model, use_ollama=True)
        self._set_status(self.t("applied_settings", model=self.settings.model))
        self.root.after(100, self._load_model_async)

    def _source_label(self) -> str:
        key = {"mic": "source_mic", "system": "source_system",
               "both": "source_both"}.get(self.settings.source, "source_mic")
        label = self.t(key)
        lang_name = next((name for c, name, _ in LANGUAGES
                          if c == self.settings.language), self.settings.language)
        label += f"   ·   🌐 {lang_name}"
        if self.settings.live_ai_correct:
            label += "   ·   ⚡ AI live"
        elif self.settings.auto_correct:
            label += "   ·   ✨ AI"
        return label


# ============================================================
# OKNO USTAWIEŃ
# ============================================================

class SettingsWindow(tk.Toplevel):
    def __init__(self, parent, settings: Settings, on_save, t_func):
        super().__init__(parent)
        self.settings = settings
        self.on_save = on_save
        self.t = t_func

        self.title(self.t("settings_title"))
        self.geometry(f"{px(560)}x{px(940)}")
        self.minsize(px(500), px(800))
        self.transient(parent)
        self.grab_set()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_ui()

    def _build_ui(self):
        p = 14
        ttk.Label(self, text=self.t("settings_title"),
                  font=("", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=p, pady=(12, 6))

        body = ttk.Frame(self, padding=(p, 0))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(13, weight=1)

        row = 0

        # Język aplikacji
        ttk.Label(body, text=self.t("app_language_label"),
                  font=("", 10, "bold")).grid(
            row=row, column=0, sticky="w", pady=(8, 2)); row += 1
        self.app_lang_var = tk.StringVar(value=self.settings.app_language)
        app_lang_frame = ttk.Frame(body)
        app_lang_frame.grid(row=row, column=0, sticky="w", padx=8); row += 1
        for code, name, flag_emoji in APP_LANGUAGES:
            ttk.Radiobutton(app_lang_frame, text=f"{flag_emoji}  {name}",
                            variable=self.app_lang_var, value=code
                            ).pack(side=tk.LEFT, padx=(0, 12))

        # Źródło
        ttk.Label(body, text=self.t("source_label"),
                  font=("", 10, "bold")).grid(
            row=row, column=0, sticky="w", pady=(10, 2)); row += 1
        self.source_var = tk.StringVar(value=self.settings.source)
        for value, key in [("mic", "source_mic"), ("system", "source_system"),
                           ("both", "source_both")]:
            ttk.Radiobutton(body, text=self.t(key), variable=self.source_var,
                            value=value).grid(row=row, column=0,
                                              sticky="w", padx=8); row += 1

        # Filtry
        ttk.Label(body, text=self.t("filters_label"),
                  font=("", 10, "bold")).grid(
            row=row, column=0, sticky="w", pady=(10, 2)); row += 1
        self.crowd_var = tk.BooleanVar(value=self.settings.ignore_crowd)
        self.noise_var = tk.BooleanVar(value=self.settings.ignore_background_noise)
        ttk.Checkbutton(body, text=self.t("filter_crowd"),
                        variable=self.crowd_var).grid(
            row=row, column=0, sticky="w", padx=8); row += 1
        ttk.Checkbutton(body, text=self.t("filter_noise"),
                        variable=self.noise_var).grid(
            row=row, column=0, sticky="w", padx=8); row += 1
        ttk.Label(body, text=self.t("filters_info"), foreground="#666",
                  justify="left").grid(
            row=row, column=0, sticky="w", padx=8, pady=(2, 6)); row += 1

        # AI
        ttk.Label(body, text=self.t("ai_label"),
                  font=("", 10, "bold")).grid(
            row=row, column=0, sticky="w", pady=(10, 2)); row += 1
        self.auto_correct_var = tk.BooleanVar(value=self.settings.auto_correct)
        ttk.Checkbutton(body, text=self.t("ai_auto"),
                        variable=self.auto_correct_var).grid(
            row=row, column=0, sticky="w", padx=8); row += 1
        self.live_ai_var = tk.BooleanVar(value=self.settings.live_ai_correct)
        ttk.Checkbutton(body, text=self.t("ai_live"),
                        variable=self.live_ai_var).grid(
            row=row, column=0, sticky="w", padx=8); row += 1

        # --- DROPDOWN MODELU OLLAMA ---
        ttk.Label(body, text=self.t("ai_model_label"), foreground="#666"
                  ).grid(row=row, column=0, sticky="w", padx=8,
                         pady=(8, 2)); row += 1

        model_row = ttk.Frame(body)
        model_row.grid(row=row, column=0, sticky="ew", padx=8); row += 1
        model_row.columnconfigure(0, weight=1)

        self.model_var = tk.StringVar(value=self.settings.postprocess_model)
        self.model_combo = ttk.Combobox(
            model_row, textvariable=self.model_var,
            state="readonly", width=30, font=("", 11),
        )
        self.model_combo.grid(row=0, column=0, sticky="ew")

        ttk.Button(model_row, text="🔄", width=3,
                   command=self._refresh_ollama_models
                   ).grid(row=0, column=1, padx=(4, 0))

        self.ollama_status_lbl = ttk.Label(
            body, text="", foreground="#888", font=("", 9),
            justify="left", wraplength=px(420))
        self.ollama_status_lbl.grid(row=row, column=0, sticky="w",
                                    padx=8, pady=(2, 6)); row += 1

        # Wypełnij listę po chwili (żeby nie blokować UI)
        self.after(50, self._refresh_ollama_models)

        # Pauza LIVE
        ttk.Label(body, text=self.t("live_silence_label"),
                  font=("", 10, "bold")).grid(
            row=row, column=0, sticky="w", pady=(10, 2)); row += 1
        sil_frame = ttk.Frame(body)
        sil_frame.grid(row=row, column=0, sticky="w", padx=8); row += 1
        self.silence_var = tk.IntVar(value=self.settings.live_silence_ms)
        self.silence_scale = ttk.Scale(
            sil_frame, from_=300, to=2000, orient=tk.HORIZONTAL, length=300,
            variable=self.silence_var,
            command=lambda v: self._silence_changed())
        self.silence_scale.pack(side=tk.LEFT)
        self.silence_lbl = ttk.Label(
            sil_frame, text=f"{self.settings.live_silence_ms} ms")
        self.silence_lbl.pack(side=tk.LEFT, padx=8)
        ttk.Label(body, text=self.t("live_silence_hint"),
                  foreground="#888", justify="left").grid(
            row=row, column=0, sticky="w", padx=8, pady=(2, 6)); row += 1

        # Język transkrypcji
        ttk.Label(body, text=self.t("transcription_language_label"),
                  font=("", 10, "bold")).grid(
            row=row, column=0, sticky="nw", pady=(10, 2)); row += 1
        lang_wrap = ttk.Frame(body)
        lang_wrap.grid(row=row, column=0, sticky="nsew", padx=8)
        lang_wrap.columnconfigure(0, weight=1)
        lang_wrap.rowconfigure(0, weight=1)
        self.lang_listbox = tk.Listbox(
            lang_wrap, height=6, exportselection=False,
            font=("", 11), activestyle="none")
        self.lang_listbox.grid(row=0, column=0, sticky="nsew")
        lang_sb = ttk.Scrollbar(lang_wrap, command=self.lang_listbox.yview)
        lang_sb.grid(row=0, column=1, sticky="ns")
        self.lang_listbox.config(yscrollcommand=lang_sb.set)
        self._lang_codes = []
        selected_idx = 0
        for idx, (code, name, country) in enumerate(LANGUAGES):
            label = (f"🌐  {name}" if code == "auto"
                     else f"{flag(country)}  {name}")
            self.lang_listbox.insert(tk.END, label)
            self._lang_codes.append(code)
            if code == self.settings.language:
                selected_idx = idx
        self.lang_listbox.selection_set(selected_idx)
        self.lang_listbox.see(selected_idx)
        row += 1

        # Model Whisper
        ttk.Label(body, text=self.t("whisper_model_label"),
                  font=("", 10, "bold")).grid(
            row=row, column=0, sticky="w", pady=(10, 2)); row += 1
        self.whisper_model_var = tk.StringVar(value=self.settings.model)
        for value, key in [("tiny", "whisper_tiny"),
                           ("base", "whisper_base"),
                           ("small", "whisper_small")]:
            ttk.Radiobutton(body, text=self.t(key),
                            variable=self.whisper_model_var,
                            value=value).grid(
                row=row, column=0, sticky="w", padx=8); row += 1

        # Przyciski
        btns = ttk.Frame(self, padding=(p, 10))
        btns.grid(row=2, column=0, sticky="ew")
        btns.columnconfigure(0, weight=1)
        ttk.Button(btns, text=self.t("reset"), command=self._reset
                   ).grid(row=0, column=0, sticky="w")
        ttk.Button(btns, text=self.t("cancel"), command=self.destroy
                   ).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text=self.t("save_btn"), command=self._save
                   ).grid(row=0, column=2)

    # ---------- Ollama dropdown ----------

    def _refresh_ollama_models(self):
        try:
            from postprocess import TextPostProcessor as _TPP
            tpp = _TPP(model=self.settings.postprocess_model, use_ollama=True)
            ok, info = tpp.ollama_status()
        except Exception as e:
            ok, info = False, f"Błąd: {e}"
            tpp = None

        if ok and tpp is not None:
            models = tpp.list_ollama_models()
            if not models:
                models = [self.settings.postprocess_model]
            self.model_combo["values"] = models
            current = self.settings.postprocess_model
            if current in models:
                self.model_var.set(current)
            elif current and any(m.startswith(current + ":") for m in models):
                match = next(m for m in models if m.startswith(current + ":"))
                self.model_var.set(match)
            else:
                self.model_var.set(models[0])
            self.ollama_status_lbl.config(
                text=f"✅ Ollama działa · {len(models)} model(i)",
                foreground="#2E7D32")
        else:
            if HAS_LLAMA_CPP and HAS_HF:
                self.model_combo["values"] = [self.settings.postprocess_model]
                self.model_var.set(self.settings.postprocess_model)
                self.ollama_status_lbl.config(
                    text="ℹ️  Ollama nie działa. Użyję lokalnego modelu z "
                         "Hugging Face (pobrany automatycznie).",
                    foreground="#1976D2")
            else:
                self.model_combo["values"] = [self.settings.postprocess_model]
                self.model_var.set(self.settings.postprocess_model)
                self.ollama_status_lbl.config(
                    text=f"⚠️  {info}", foreground="#C62828")

    def _silence_changed(self):
        self.silence_lbl.config(text=f"{int(self.silence_var.get())} ms")

    # ---------- Save / Reset ----------

    def _save(self):
        old_lang = self.settings.app_language
        sel = self.lang_listbox.curselection()
        if sel:
            self.settings.language = self._lang_codes[sel[0]]

        self.settings.app_language = self.app_lang_var.get()
        self.settings.source = self.source_var.get()
        self.settings.ignore_crowd = self.crowd_var.get()
        self.settings.ignore_background_noise = self.noise_var.get()
        self.settings.auto_correct = self.auto_correct_var.get()
        self.settings.live_ai_correct = self.live_ai_var.get()

        chosen = (self.model_var.get() or "").strip()
        if chosen:
            self.settings.postprocess_model = chosen

        self.settings.live_silence_ms = int(self.silence_var.get())
        self.settings.model = self.whisper_model_var.get()
        self.settings.save()
        self.on_save()

        if old_lang != self.settings.app_language:
            from i18n import translate as tr
            messagebox.showinfo(
                tr("info", self.settings.app_language),
                tr("language_changed", self.settings.app_language))
        self.destroy()

    def _reset(self):
        if not messagebox.askyesno(self.t("reset"), self.t("reset") + "?"):
            return
        new = Settings.reset()
        for k in self.settings.__dataclass_fields__:
            setattr(self.settings, k, getattr(new, k))

        self.app_lang_var.set(new.app_language)
        self.source_var.set(new.source)
        self.crowd_var.set(new.ignore_crowd)
        self.noise_var.set(new.ignore_background_noise)
        self.auto_correct_var.set(new.auto_correct)
        self.live_ai_var.set(new.live_ai_correct)
        self.silence_var.set(new.live_silence_ms)
        self.silence_lbl.config(text=f"{new.live_silence_ms} ms")
        self.whisper_model_var.set(new.model)
        self.after(50, self._refresh_ollama_models)

        self.lang_listbox.selection_clear(0, tk.END)
        for idx, code in enumerate(self._lang_codes):
            if code == new.language:
                self.lang_listbox.selection_set(idx)
                self.lang_listbox.see(idx)
                break
        self.on_save()
        self.destroy()


# ============================================================
# OKNO HISTORII
# ============================================================

class HistoryWindow(tk.Toplevel):
    def __init__(self, parent, settings):
        super().__init__(parent)
        from i18n import translate
        self.t = lambda k, **kw: translate(k, settings.app_language, **kw)
        self.title(self.t("history_title"))
        self.geometry(f"{px(640)}x{px(480)}")
        self.transient(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        wrap = ttk.Frame(self, padding=10)
        wrap.grid(row=0, column=0, sticky="nsew")
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(wrap, font=("", 11), activestyle="none")
        self.listbox.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(wrap, command=self.listbox.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        preview_wrap = ttk.Frame(self, padding=(10, 0, 10, 10))
        preview_wrap.grid(row=1, column=0, sticky="ew")
        preview_wrap.columnconfigure(0, weight=1)
        self.preview = ScrolledText(preview_wrap, height=8, wrap=tk.WORD,
                                    font=("", 11))
        self.preview.grid(row=0, column=0, sticky="ew")
        self._entries = []
        self._load()

    def _load(self):
        try:
            from storage import load_sessions
            sessions = load_sessions()
            if not sessions:
                self.listbox.insert(tk.END, self.t("no_history"))
                return
            self._entries = sessions
            for s in sessions:
                ts = s.display_date()
                model = s.model or "?"
                lang = s.language or "?"
                preview = (s.text or "")[:60].replace("\n", " ")
                self.listbox.insert(tk.END, f"[{ts}] ({model}/{lang})  {preview}…")
        except Exception as e:
            self.listbox.insert(tk.END, f"(error: {e})")

    def _on_select(self, _event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._entries):
            return
        text = self._entries[idx].text or ""
        self.preview.delete("1.0", tk.END)
        self.preview.insert("1.0", text)


# ============================================================
# START
# ============================================================

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()