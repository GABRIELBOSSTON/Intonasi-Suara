"""CPU tracker adapters; no edits to the production pipeline."""
import numpy as np
import parselmouth
from generate import SR, in_regions

NAMES = ["praat_current", "praat_hirst", "praat_hirst_cost04", "praat_post",
         "pyin", "torchcrepe_tiny", "swift_f0", "torchcrepe_full"]
DEFAULT_NAMES = NAMES[:-1]


def reject_outliers(f0, window=11):
    """Reject isolated octave-scale excursions; never fill unvoiced gaps."""
    output = np.asarray(f0, float).copy()
    raw = output.copy()
    rejected = np.zeros(len(raw), bool)
    for i, value in enumerate(raw):
        if not np.isfinite(value) or value <= 0:
            continue
        neighbors = raw[max(0, i - window // 2):min(len(raw), i + window // 2 + 1)]
        neighbors = neighbors[np.isfinite(neighbors) & (neighbors > 0)]
        if len(neighbors) < 5:
            continue
        median = np.median(neighbors)
        if abs(12 * np.log2(value / median)) > 8:
            output[i] = np.nan
            rejected[i] = True
    return output, rejected


class Trackers:
    def __init__(self, cfg, threads=4):
        self.cfg = cfg
        self.threads = threads
        self.swift = None

    def run(self, name, audio, segments):
        if name.startswith("praat"):
            return self.praat(name, audio, segments)
        if name == "pyin":
            import librosa
            f0, voiced, probability = librosa.pyin(audio.astype(float), fmin=50, fmax=600,
                sr=SR, frame_length=1024, hop_length=160, fill_na=np.nan)
            f0[~voiced] = np.nan
            return np.arange(len(f0)) * .01, f0, {"mean_probability": float(np.mean(probability))}
        if name in ("torchcrepe_full", "torchcrepe_tiny"):
            import torch
            import torchcrepe
            # torchcrepe adds triangular dither through scipy/NumPy.
            np.random.seed(20261026)
            torch.manual_seed(20261026)
            with torch.inference_mode():
                tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
                pitch, periodicity = torchcrepe.predict(tensor, SR, 160, 50, 600, name.rsplit("_", 1)[1],
                    batch_size=512, device="cpu", return_periodicity=True)
                periodicity = torchcrepe.filter.median(periodicity, 3)
                periodicity = torchcrepe.threshold.Silence(-60)(periodicity, tensor, SR, 160)
                f0 = pitch[0].numpy().astype(float)
                f0[periodicity[0].numpy() < .21] = np.nan
            return np.arange(len(f0)) * .01, f0, {"periodicity_threshold": .21}
        if name == "swift_f0":
            from swift_f0 import SwiftF0
            if self.swift is None:
                self.swift = SwiftF0(threads=self.threads, spin=False)
            result = self.swift.detect(audio, SR, fmin=50, fmax=600)
            f0 = result.pitch_hz.copy()
            f0[result.confidence < .5] = np.nan
            return result.timestamps, f0, {"confidence_threshold": .5}
        raise ValueError(f"Unknown tracker: {name}")

    def praat(self, name, audio, segments):
        cfg = self.cfg
        masked = np.where(in_regions(np.arange(len(audio)) / SR, segments), audio, 0)
        sound = parselmouth.Sound(masked, sampling_frequency=SR)
        cost = .04 if name == "praat_hirst_cost04" else .01
        def track(floor, ceiling):
            return sound.to_pitch_ac(time_step=cfg.F0_STEP_SECONDS, pitch_floor=floor,
                pitch_ceiling=ceiling, silence_threshold=cfg.F0_SILENCE_THRESHOLD,
                voicing_threshold=cfg.F0_VOICING_THRESHOLD, octave_cost=cost,
                octave_jump_cost=.35, voiced_unvoiced_cost=.14)
        first = track(cfg.F0_FLOOR_HZ, cfg.F0_CEILING_HZ)
        f0 = first.selected_array["frequency"]
        valid = (f0 > 0) & in_regions(first.xs(), segments)
        floor, ceiling = cfg.F0_FLOOR_HZ, cfg.F0_CEILING_HZ
        if valid.sum() >= cfg.F0_MIN_ADAPT_FRAMES:
            low, high = np.percentile(f0[valid], [cfg.F0_ADAPT_LOW_PERCENTILE, cfg.F0_ADAPT_HIGH_PERCENTILE])
            floor = max(floor, low * .75 if name.startswith("praat_hirst") else low / cfg.F0_ADAPT_MARGIN)
            ceiling = min(ceiling, high * cfg.F0_ADAPT_MARGIN)
        result = track(floor, ceiling)
        f0 = result.selected_array["frequency"].astype(float)
        f0[(f0 <= 0) | ~in_regions(result.xs(), segments)] = np.nan
        meta = {"floor": float(floor), "ceiling": float(ceiling), "octave_cost": cost,
                "rejected_pct": 0.0}
        if name == "praat_post":
            count = np.isfinite(f0).sum()
            f0, rejected = reject_outliers(f0)
            meta["rejected_pct"] = float(100 * rejected.sum() / count) if count else 0.
        return result.xs(), f0, meta
