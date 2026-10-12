"""Transkrypcja Whisper – plikowa i strumieniowa z segmentacją po pauzach."""
import queue
import sys
import tempfile
import threading
import wave

import numpy as np

SYSTEM = sys.platform

if SYSTEM == "darwin":
    from vad import VADProcessor
else:
    VADProcessor = None

try:
    from silero_vad_lite import SileroVAD
    _HAS_SILERO = True
except ImportError:
    _HAS_SILERO = False

if SYSTEM == "darwin":
    import mlx_whisper
else:
    from faster_whisper import WhisperModel


VAD_THRESHOLDS = {"none": None, "noise": 0.6, "crowd": 0.75}

VAD_PARAMS = {
    "none": None,
    "noise": {
        "threshold": 0.6,
        "min_speech_duration_ms": 300,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
    },
    "crowd": {
        "threshold": 0.75,
        "min_speech_duration_ms": 500,
        "min_silence_duration_ms": 800,
        "speech_pad_ms": 200,
    },
}


class UtteranceSegmenter:
    FRAME_SIZE = 512

    def __init__(self, sample_rate: int = 16000,
                 speech_threshold: float = 0.3,
                 min_speech_ms: int = 250,
                 min_silence_ms: int = 700,
                 max_utterance_sec: float = 20.0):
        if not _HAS_SILERO:
            raise RuntimeError("Brak 'silero-vad-lite'.")
        self.vad = SileroVAD(sample_rate)
        self.speech_threshold = speech_threshold
        self.min_speech_frames = max(1, int(min_speech_ms / 32))
        self.min_silence_frames = max(1, int(min_silence_ms / 32))
        self.max_frames = int(max_utterance_sec * 1000 / 32)
        self._reset()

    def _reset(self):
        self.buffer: list[np.ndarray] = []
        self.speech_frames = 0
        self.silence_frames = 0
        self.in_speech = False

    def set_silence_ms(self, ms: int):
        self.min_silence_frames = max(1, int(ms / 32))

    def process(self, chunk: np.ndarray) -> list[np.ndarray]:
        completed: list[np.ndarray] = []
        for i in range(0, len(chunk) - self.FRAME_SIZE + 1, self.FRAME_SIZE):
            frame = chunk[i:i + self.FRAME_SIZE]
            try:
                prob = float(self.vad.process(frame))
            except Exception:
                prob = 0.0
            self.buffer.append(frame)
            if prob >= self.speech_threshold:
                self.speech_frames += 1
                self.silence_frames = 0
                self.in_speech = True
            else:
                if self.in_speech:
                    self.silence_frames += 1
                    if self.silence_frames >= self.min_silence_frames:
                        utt = self._finalize()
                        if utt is not None:
                            completed.append(utt)
            if (self.in_speech and len(self.buffer) >= self.max_frames
                    and self.speech_frames >= self.min_speech_frames):
                utt = self._finalize()
                if utt is not None:
                    completed.append(utt)
        return completed

    def flush(self) -> np.ndarray | None:
        if self.in_speech and self.speech_frames >= self.min_speech_frames:
            utt = self._finalize()
            self._reset()
            return utt
        self._reset()
        return None

    def _finalize(self) -> np.ndarray | None:
        if self.speech_frames < self.min_speech_frames or not self.buffer:
            self._reset()
            return None
        audio = np.concatenate(self.buffer).astype(np.float32)
        self._reset()
        return audio


class Transcriber:
    FALLBACKS = ["base", "tiny"]

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> bool:
        if SYSTEM == "darwin":
            return True
        with self._lock:
            if self.model is not None:
                return True
            candidates = [self.model_name] + [m for m in self.FALLBACKS
                                              if m != self.model_name]
            last_error = None
            for name in candidates:
                try:
                    print(f"[transcriber] Ładuję model: {name} …", flush=True)
                    self.model = WhisperModel(name, device="cpu", compute_type="int8")
                    self.model_name = name
                    self._load_error = None
                    print(f"[transcriber] Model '{name}' gotowy.", flush=True)
                    return True
                except Exception as e:
                    last_error = f"{name}: {e}"
                    print(f"[transcriber] Błąd '{name}': {e}", flush=True)
            self._load_error = f"Nie udało się załadować modelu.\nOstatni błąd: {last_error}"
            return False

    def _vad_key(self, ignore_crowd: bool, ignore_noise: bool) -> str:
        if ignore_crowd:
            return "crowd"
        if ignore_noise:
            return "noise"
        return "none"

    def _vad_threshold(self, ignore_crowd: bool, ignore_noise: bool):
        return VAD_THRESHOLDS[self._vad_key(ignore_crowd, ignore_noise)]

    def _vad_params_win(self, ignore_crowd: bool, ignore_noise: bool):
        return VAD_PARAMS[self._vad_key(ignore_crowd, ignore_noise)]

    def _filter_with_vad_macos(self, audio: np.ndarray, threshold: float):
        if VADProcessor is None:
            return audio
        vad = VADProcessor(sample_rate=16000, threshold=threshold)
        segments = vad.process_chunk(audio)
        segments.extend(vad.flush())
        if not segments:
            return None
        return np.concatenate(segments).astype(np.float32)

    def transcribe(self, wav_path: str, language: str = "auto",
                   ignore_crowd: bool = False, ignore_noise: bool = False,
                   initial_prompt: str | None = None) -> str:
        lang = None if language == "auto" else language
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            samp_width = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        if samp_width != 2:
            raise RuntimeError("Obsługiwany tylko 16-bit WAV.")
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        return self._transcribe_audio(audio, lang, ignore_crowd, ignore_noise, initial_prompt)

    def _transcribe_audio(self, audio: np.ndarray, lang,
                          ignore_crowd: bool, ignore_noise: bool,
                          initial_prompt: str | None = None) -> str:
        if SYSTEM == "darwin":
            thr = self._vad_threshold(ignore_crowd, ignore_noise)
            if thr is not None:
                filtered = self._filter_with_vad_macos(audio, thr)
                if filtered is None:
                    return ""
                audio = filtered

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_i16.tobytes())

        if SYSTEM == "darwin":
            kwargs = dict(
                path_or_hf_repo=f"mlx-community/whisper-{self.model_name}-mlx",
                language=lang, task="transcribe", fp16=True,
            )
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt
            result = mlx_whisper.transcribe(tmp_path, **kwargs)
            return result.get("text", "").strip()

        if self.model is None:
            raise RuntimeError("Model nie został załadowany.")
        kwargs = {"language": lang, "task": "transcribe"}
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        vp = self._vad_params_win(ignore_crowd, ignore_noise)
        if vp is not None:
            kwargs["vad_filter"] = True
            kwargs["vad_parameters"] = vp
        segments, _ = self.model.transcribe(tmp_path, **kwargs)
        return "".join(s.text for s in segments).strip()

    def transcribe_stream(self, audio_queue, language="auto", on_text=None,
                          on_error=None, stop_event=None, ignore_crowd=False,
                          ignore_noise=False, silence_ms=700):
        lang = None if language == "auto" else language
        if SYSTEM != "darwin" and self.model is None:
            if on_error:
                on_error("Model nie został załadowany.")
            return
        try:
            segmenter = UtteranceSegmenter(min_silence_ms=silence_ms)
        except Exception as e:
            if on_error:
                on_error(f"Błąd VAD: {e}")
            return

        accumulated_text = ""
        MAX_CONTEXT = 200

        def emit(audio):
            nonlocal accumulated_text
            if len(audio) < 1600:
                return
            try:
                text = self._transcribe_audio(
                    audio, lang, ignore_crowd, ignore_noise,
                    initial_prompt=accumulated_text[-MAX_CONTEXT:] or None,
                )
                if text:
                    accumulated_text = (accumulated_text + " " + text).strip()
                    if on_text:
                        on_text(text)
            except Exception as e:
                if on_error:
                    on_error(str(e))

        while not (stop_event and stop_event.is_set()):
            try:
                chunk = audio_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if chunk is None:
                break
            try:
                for utt in segmenter.process(chunk):
                    emit(utt)
            except Exception as e:
                if on_error:
                    on_error(str(e))

        try:
            last = segmenter.flush()
            if last is not None:
                emit(last)
        except Exception as e:
            if on_error:
                on_error(str(e))