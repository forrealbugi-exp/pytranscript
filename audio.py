"""Nagrywanie dźwięku: plikowe i na żywo (małe kawałki)."""
import queue
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import soundcard as sc

from storage import get_app_dir

SAMPLE_RATE = 16000
RECORDINGS_DIR = get_app_dir() / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True, parents=True)


class AudioRecorder:
    def __init__(self, source: str = "mic"):
        self.source = source
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.mic_frames: list[np.ndarray] = []
        self.system_frames: list[np.ndarray] = []
        self.errors: list[str] = []

    def start(self) -> None:
        self.stop_event.clear()
        self.mic_frames = []
        self.system_frames = []
        self.errors = []
        self.threads = []
        if self.source in ("mic", "both"):
            t = threading.Thread(target=self._record_mic, daemon=True)
            t.start()
            self.threads.append(t)
        if self.source in ("system", "both"):
            t = threading.Thread(target=self._record_system, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self) -> str | None:
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=2)
        mic = np.concatenate(self.mic_frames) if self.mic_frames else np.array([], dtype=np.float32)
        sys_a = np.concatenate(self.system_frames) if self.system_frames else np.array([], dtype=np.float32)
        audio = self._mix(mic, sys_a)
        if audio.size == 0:
            return None
        return self._save_wav(audio)

    def _record_mic(self) -> None:
        try:
            mic = sc.default_microphone()
            if mic is None:
                self.errors.append("Nie znaleziono mikrofonu.")
                return
            with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
                while not self.stop_event.is_set():
                    data = rec.record(numframes=int(SAMPLE_RATE * 0.1))
                    if data.ndim > 1:
                        data = data[:, 0]
                    self.mic_frames.append(data.astype(np.float32))
        except Exception as e:
            self.errors.append(f"Błąd mikrofonu: {e}")

    def _record_system(self) -> None:
        try:
            speaker = sc.default_speaker()
            if speaker is None:
                self.errors.append("Nie znaleziono urządzenia wyjściowego.")
                return
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            if loopback is None:
                candidates = [m for m in sc.all_microphones(include_loopback=False)
                              if "BlackHole" in m.name or "Loopback" in m.name]
                if not candidates:
                    self.errors.append("Brak urządzenia do przechwytywania dźwięku systemowego.")
                    return
                loopback = candidates[0]
            with loopback.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
                while not self.stop_event.is_set():
                    data = rec.record(numframes=int(SAMPLE_RATE * 0.1))
                    if data.ndim > 1:
                        data = data[:, 0]
                    self.system_frames.append(data.astype(np.float32))
        except Exception as e:
            self.errors.append(f"Błąd dźwięku systemowego: {e}")

    def _mix(self, mic: np.ndarray, system: np.ndarray) -> np.ndarray:
        if self.source == "mic":
            return mic
        if self.source == "system":
            return system
        if mic.size and system.size:
            n = min(mic.size, system.size)
            return mic[:n] * 0.5 + system[:n] * 0.5
        return mic if mic.size else system

    def _save_wav(self, audio: np.ndarray) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = RECORDINGS_DIR / f"nagranie_{timestamp}.wav"
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        try:
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())
        except Exception as e:
            raise RuntimeError(f"Nie udało się zapisać WAV: {e}") from e
        return str(wav_path)


class LiveAudioRecorder:
    def __init__(self, source: str = "mic", chunk_seconds: float = 0.1):
        self.source = source
        self.chunk_seconds = chunk_seconds
        self.chunk_samples = int(SAMPLE_RATE * chunk_seconds)
        self.queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.errors: list[str] = []

    def start(self) -> None:
        self.stop_event.clear()
        self.errors = []
        self.threads = []
        if self.source in ("mic", "both"):
            t = threading.Thread(target=self._loop_mic, daemon=True)
            t.start()
            self.threads.append(t)
        if self.source in ("system", "both"):
            t = threading.Thread(target=self._loop_system, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self) -> None:
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=2)
        self.queue.put(None)

    def _push(self, chunk: np.ndarray) -> None:
        chunk = np.clip(chunk.flatten(), -1.0, 1.0).astype(np.float32)
        if chunk.size:
            self.queue.put(chunk)

    def _loop_mic(self) -> None:
        try:
            mic = sc.default_microphone()
            if mic is None:
                self.errors.append("Nie znaleziono mikrofonu.")
                return
            with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
                while not self.stop_event.is_set():
                    data = rec.record(numframes=self.chunk_samples)
                    if data.ndim > 1:
                        data = data[:, 0]
                    self._push(data)
        except Exception as e:
            self.errors.append(f"Błąd mikrofonu: {e}")

    def _loop_system(self) -> None:
        try:
            speaker = sc.default_speaker()
            if speaker is None:
                self.errors.append("Nie znaleziono urządzenia wyjściowego.")
                return
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            if loopback is None:
                candidates = [m for m in sc.all_microphones(include_loopback=False)
                              if "BlackHole" in m.name or "Loopback" in m.name]
                if not candidates:
                    self.errors.append("Brak urządzenia do przechwytywania dźwięku systemowego.")
                    return
                loopback = candidates[0]
            with loopback.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
                while not self.stop_event.is_set():
                    data = rec.record(numframes=self.chunk_samples)
                    if data.ndim > 1:
                        data = data[:, 0]
                    self._push(data)
        except Exception as e:
            self.errors.append(f"Błąd dźwięku systemowego: {e}")