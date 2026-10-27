"""Central configuration; thresholds are unvalidated pilot hypotheses."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Literal

ANALYSIS_VERSION = "0.5.1"
# ekspresivitas = nada (zona wajar) + tempo bicara; intonasi = nada saja.
MODE: Literal["intonasi", "ekspresivitas"] = "ekspresivitas"
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    ANALYSIS_VERSION: str = ANALYSIS_VERSION
    MODE: Literal["intonasi", "ekspresivitas"] = MODE
    SAMPLE_RATE: int = 16000
    FFMPEG: str = "ffmpeg"
    FFPROBE: str = "ffprobe"
    FFMPEG_TIMEOUT_SECONDS: float = 120.0
    MAX_INPUT_MB: float = 512.0
    MAX_DURATION_SECONDS: float = 600.0
    MAX_DECODED_SAMPLES: int = 60_000_000
    MIN_AUDIO_SECONDS: float = 1.0
    # Bukan aturan kebijakan: hanya batas teknis agar pitch/tempo bisa dihitung.
    MIN_SPEECH_SECONDS: float = 3.0
    CLIP_AMPLITUDE: float = 0.999
    CLIP_WARN_PCT: float = 0.1
    CLIP_REVIEW_PCT: float = 1.0
    CHANNEL_WEAK_RATIO: float = 0.05
    CHANNEL_ANTIPHASE_CORRELATION: float = -0.8
    SNR_WARN_DB: float = 20.0
    SNR_LOW_DB: float = 10.0
    QC_FRAME_SECONDS: float = 0.02
    NOISE_MARGIN_SECONDS: float = 0.1
    MIN_NOISE_SECONDS: float = 0.5
    MIN_SNR_SPEECH_SECONDS: float = 0.5
    POWER_EPSILON: float = 1e-12
    VAD_THRESHOLD: float = 0.5
    VAD_MIN_SPEECH_MS: int = 100
    VAD_MIN_SILENCE_MS: int = 100
    VAD_SPEECH_PAD_MS: int = 0
    PAUSE_MIN_SECONDS: float = 0.25
    LONG_PAUSE_SECONDS: float = 2.0
    # CREPE tiny lebih tahan dengung/bising latar daripada Praat raw AC pada
    # rekaman HP (lihat research/riset-pitch-tracking.md). Praat tetap dihitung
    # sebagai pembanding audit, tidak memengaruhi keputusan.
    PITCH_TRACKER: str = "crepe"
    PITCH_AUDIT_PRAAT: bool = True
    CREPE_MODEL: str = "tiny"
    CREPE_PERIODICITY_THRESHOLD: float = 0.21
    # Relatif terhadap P90 energi frame bicara orang itu sendiri (tidak
    # bergantung volume rekaman).
    CREPE_RELATIVE_SILENCE_DB: float = -40.0
    CREPE_BATCH_SIZE: int = 512
    CREPE_SEED: int = 20261026
    # Sertakan kontur F0 per frame di hasil (untuk alat tes visual); CLI default mati agar JSON ringkas.
    PITCH_INCLUDE_CONTOUR: bool = False
    F0_FLOOR_HZ: float = 50.0
    F0_CEILING_HZ: float = 600.0
    F0_STEP_SECONDS: float = 0.01
    F0_VOICING_THRESHOLD: float = 0.45
    F0_SILENCE_THRESHOLD: float = 0.03
    F0_ADAPT_MARGIN: float = 1.5
    # Rentang pass 2 dari kuartil (gaya De Looze & Hirst) agar salah oktaf di
    # pass 1 tidak menurunkan floor sampai 50 Hz.
    F0_ADAPT_LOW_PERCENTILE: float = 25.0
    F0_ADAPT_HIGH_PERCENTILE: float = 75.0
    F0_RANGE_LOW_PERCENTILE: float = 10.0
    F0_RANGE_HIGH_PERCENTILE: float = 90.0
    F0_MIN_ADAPT_FRAMES: int = 20
    MIN_PHRASE_VOICED_SECONDS: float = 0.3
    MIN_F0_COVERAGE: float = 0.3
    OCTAVE_JUMP_ST: float = 9.0
    MAX_OCTAVE_JUMPS_PER_MIN: float = 5.0
    OCTAVE_REVIEW_MIN_COUNT: int = 3
    MIN_PITCH_RANGE_ST: float = 4.0
    MIN_PITCH_SD_ST: float = 2.0
    PITCH_RANGE_BORDER_ST: float = 0.25
    PITCH_SD_BORDER_ST: float = 0.15
    # Batas atas zona wajar: di atas ini biasanya salah ukur, jadi ditinjau manual.
    MAX_PITCH_RANGE_ST: float = 15.0
    MAX_PITCH_SD_ST: float = 7.0
    # Tempo (mode ekspresivitas). Hipotesis pilot dari 4 video sampel, belum tervalidasi.
    MIN_ARTICULATION_SYL_S: float = 4.5
    ARTICULATION_BORDER_SYL_S: float = 0.25
    FAST_ARTICULATION_SYL_S: float = 9.0
    ASR_MODEL_SIZE: str = "small"
    ASR_REPO: str = "Systran/faster-whisper-small"
    ASR_REVISION: str = "536b0662742c02347bc0e980a01041f333bce120"
    ASR_LOCAL_PATH: str = ""
    ASR_CACHE_DIR: str = str(ROOT / ".cache" / "models")
    ASR_LOCAL_FILES_ONLY: bool = False
    ASR_CPU_THREADS: int = 4
    ASR_COMPUTE_TYPE: str = "int8"
    ASR_BEAM_SIZE: int = 5
    ASR_CONDITION_ON_PREVIOUS_TEXT: bool = False
    ASR_ENABLED: bool = True
    EVIDENCE_LIMIT: int = 3
    VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".3gp")

    def __post_init__(self):
        # Validate both direct construction and JSON overrides before loading models.
        signed = {"CHANNEL_ANTIPHASE_CORRELATION", "SNR_WARN_DB", "SNR_LOW_DB", "CREPE_RELATIVE_SILENCE_DB"}
        for field in fields(self):
            value, default = getattr(self, field.name), field.default
            if isinstance(default, tuple):
                valid_type = isinstance(value, tuple) and all(isinstance(item, str) for item in value)
            elif isinstance(default, float):
                valid_type = type(value) in (int, float)
            else:
                valid_type = type(value) is type(default)
            if not valid_type:
                raise ValueError(f"Tipe {field.name} tidak valid")
            if type(value) in (int, float):
                if not math.isfinite(value):
                    raise ValueError(f"{field.name} harus finite")
                if value < 0 and field.name not in signed:
                    raise ValueError(f"{field.name} tidak boleh negatif")
        if self.MODE not in ("intonasi", "ekspresivitas"):
            raise ValueError("MODE harus intonasi atau ekspresivitas")
        if self.SAMPLE_RATE != 16000:
            raise ValueError("Fase 1 menggunakan SAMPLE_RATE=16000")
        for name in ("FFMPEG_TIMEOUT_SECONDS", "MAX_DURATION_SECONDS", "MAX_INPUT_MB",
                     "MAX_DECODED_SAMPLES", "MIN_AUDIO_SECONDS", "MIN_SPEECH_SECONDS",
                     "MIN_NOISE_SECONDS", "MIN_SNR_SPEECH_SECONDS", "MIN_PHRASE_VOICED_SECONDS",
                     "F0_MIN_ADAPT_FRAMES", "VAD_MIN_SPEECH_MS", "VAD_MIN_SILENCE_MS",
                     "F0_STEP_SECONDS", "QC_FRAME_SECONDS", "POWER_EPSILON",
                     "PAUSE_MIN_SECONDS", "ASR_CPU_THREADS", "ASR_BEAM_SIZE"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} harus positif")
        for name in ("VAD_THRESHOLD", "MIN_F0_COVERAGE", "F0_VOICING_THRESHOLD", "CLIP_AMPLITUDE"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} harus dalam (0, 1]")
        for name in ("CHANNEL_WEAK_RATIO", "F0_SILENCE_THRESHOLD"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} harus dalam [0, 1]")
        if not -1 <= self.CHANNEL_ANTIPHASE_CORRELATION <= 0:
            raise ValueError("CHANNEL_ANTIPHASE_CORRELATION harus dalam [-1, 0]")
        if self.MIN_AUDIO_SECONDS > self.MAX_DURATION_SECONDS or self.MIN_SPEECH_SECONDS > self.MAX_DURATION_SECONDS:
            raise ValueError("Durasi minimum tidak boleh melebihi durasi maksimum")
        if self.SNR_LOW_DB > self.SNR_WARN_DB:
            raise ValueError("Ambang SNR terbalik")
        if not 0 < self.F0_FLOOR_HZ < self.F0_CEILING_HZ < self.SAMPLE_RATE / 2:
            raise ValueError("Rentang F0 tidak valid")
        if self.PITCH_TRACKER not in ("crepe", "praat"):
            raise ValueError("PITCH_TRACKER harus crepe atau praat")
        if self.CREPE_MODEL not in ("tiny", "full"):
            raise ValueError("CREPE_MODEL harus tiny atau full")
        if not 0 < self.CREPE_PERIODICITY_THRESHOLD < 1:
            raise ValueError("CREPE_PERIODICITY_THRESHOLD harus dalam (0, 1)")
        if self.CREPE_RELATIVE_SILENCE_DB >= 0 or self.CREPE_BATCH_SIZE <= 0:
            raise ValueError("CREPE_RELATIVE_SILENCE_DB harus negatif dan CREPE_BATCH_SIZE positif")
        if not (self.MIN_PITCH_RANGE_ST < self.MAX_PITCH_RANGE_ST and self.MIN_PITCH_SD_ST < self.MAX_PITCH_SD_ST):
            raise ValueError("Batas zona nada terbalik")
        if not 0 < self.MIN_ARTICULATION_SYL_S < self.FAST_ARTICULATION_SYL_S:
            raise ValueError("Ambang tempo tidak valid")
        if self.F0_ADAPT_MARGIN <= 1:
            raise ValueError("F0_ADAPT_MARGIN harus >1")
        if self.VAD_SPEECH_PAD_MS != 0 or self.VAD_MIN_SILENCE_MS >= self.PAUSE_MIN_SECONDS * 1000:
            raise ValueError("VAD pengukuran harus tanpa padding dan silence di bawah batas jeda")
        if self.LONG_PAUSE_SECONDS < self.PAUSE_MIN_SECONDS:
            raise ValueError("Batas jeda panjang harus >= batas jeda")
        if not 0 <= self.CLIP_WARN_PCT <= self.CLIP_REVIEW_PCT <= 100:
            raise ValueError("Ambang clipping terbalik")
        for low, high in ((self.F0_ADAPT_LOW_PERCENTILE, self.F0_ADAPT_HIGH_PERCENTILE),
                          (self.F0_RANGE_LOW_PERCENTILE, self.F0_RANGE_HIGH_PERCENTILE)):
            if not 0 <= low < high <= 100:
                raise ValueError("Persentil F0 tidak valid")
        for name in ("ANALYSIS_VERSION", "FFMPEG", "FFPROBE", "ASR_MODEL_SIZE", "ASR_CACHE_DIR"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} tidak boleh kosong")
        if self.ASR_COMPUTE_TYPE not in ("int8", "int8_float32", "float32", "auto"):
            raise ValueError("ASR_COMPUTE_TYPE tidak didukung konfigurasi CPU")
        if self.ASR_ENABLED and not self.ASR_LOCAL_PATH:
            if self.ASR_REPO != f"Systran/faster-whisper-{self.ASR_MODEL_SIZE}":
                raise ValueError("ASR_MODEL_SIZE dan ASR_REPO tidak cocok")
            if self.ASR_MODEL_SIZE.endswith(".en"):
                raise ValueError("Gunakan model multilingual untuk bahasa Indonesia")
            if re.fullmatch(r"[0-9a-f]{40}", self.ASR_REVISION) is None:
                raise ValueError("ASR_REVISION harus commit SHA 40 karakter")
        if not self.VIDEO_EXTENSIONS or any(re.fullmatch(r"\.[A-Za-z0-9]+", item) is None for item in self.VIDEO_EXTENSIONS):
            raise ValueError("VIDEO_EXTENSIONS harus daftar ekstensi, misalnya .mp4")
        object.__setattr__(self, "VIDEO_EXTENSIONS", tuple(dict.fromkeys(item.lower() for item in self.VIDEO_EXTENSIONS)))

    def snapshot(self):
        return asdict(self)

    def fingerprint(self):
        payload = json.dumps(self.snapshot(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def load_config(path=None):
    if path is None:
        return Config()
    def unique_keys(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"Konfigurasi duplikat: {key}")
            output[key] = value
        return output

    values = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_keys)
    if not isinstance(values, dict):
        raise ValueError("Konfigurasi harus berupa objek JSON")
    defaults = Config().snapshot()
    for key, value in values.items():
        if key not in defaults:
            raise ValueError(f"Konfigurasi tidak dikenal: {key}")
        default = defaults[key]
        if isinstance(default, tuple):
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"{key} harus array string")
            values[key] = tuple(value)
        elif isinstance(default, float):
            if type(value) not in (int, float):
                raise ValueError(f"{key} harus angka")
        elif type(value) is not type(default):
            raise ValueError(f"Tipe {key} tidak valid")
    return Config(**values)
