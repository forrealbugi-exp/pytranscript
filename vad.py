"""Detekcja mowy (VAD) z użyciem Silero VAD."""
import numpy as np

try:
    from silero_vad_lite import SileroVAD
    _HAS_SILERO = True
except ImportError:
    _HAS_SILERO = False

FRAME_SIZE = 512
FRAME_MS = 32


class VADProcessor:
    def __init__(self, sample_rate: int = 16000,
                 threshold: float = 0.5,
                 min_speech_ms: int = 250,
                 min_silence_ms: int = 400):
        if not _HAS_SILERO:
            raise RuntimeError("Brak 'silero-vad-lite'. pip install silero-vad-lite")
        self.vad = SileroVAD(sample_rate)
        self.threshold = threshold
        self.min_speech_frames = max(1, int(min_speech_ms / FRAME_MS))
        self.min_silence_frames = max(1, int(min_silence_ms / FRAME_MS))
        self._reset_state()

    def _reset_state(self):
        self.speech_frames: list[np.ndarray] = []
        self.silence_counter = 0
        self.in_speech = False

    def reset(self):
        try:
            self.vad.reset()
        except Exception:
            pass
        self._reset_state()

    def process_chunk(self, chunk: np.ndarray) -> list[np.ndarray]:
        segments = []
        for i in range(0, len(chunk) - FRAME_SIZE + 1, FRAME_SIZE):
            frame = chunk[i:i + FRAME_SIZE]
            try:
                prob = float(self.vad.process(frame))
            except Exception:
                prob = 0.0
            is_speech = prob >= self.threshold
            if is_speech:
                self.silence_counter = 0
                if not self.in_speech:
                    self.in_speech = True
                    self.speech_frames = []
                self.speech_frames.append(frame)
            else:
                if self.in_speech:
                    self.silence_counter += 1
                    self.speech_frames.append(frame)
                    if self.silence_counter >= self.min_silence_frames:
                        seg = self._finalize()
                        if seg is not None:
                            segments.append(seg)
        return segments

    def flush(self) -> list[np.ndarray]:
        seg = self._finalize()
        return [seg] if seg is not None else []

    def _finalize(self) -> np.ndarray | None:
        if not self.in_speech:
            return None
        frames = self.speech_frames
        self.in_speech = False
        self.speech_frames = []
        self.silence_counter = 0
        if len(frames) < self.min_speech_frames:
            return None
        return np.concatenate(frames).astype(np.float32)