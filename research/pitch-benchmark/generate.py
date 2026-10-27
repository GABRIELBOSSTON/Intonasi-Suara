"""Deterministic source-filter-like fixtures with explicit F0 references."""
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf

SR = 16000
STEP = .01


def in_regions(times, regions):
    mask = np.zeros(len(times), dtype=bool)
    for start, end in regions:
        mask |= (times >= start) & (times < end)
    return mask


def add_noise(audio, noise, speech, snr_db):
    gain = np.sqrt(np.mean(audio[speech] ** 2) /
                   max(np.mean(noise[speech] ** 2), 1e-12)) * 10 ** (-snr_db / 20)
    return audio + gain * noise


def make_cases(folder, ffmpeg):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cases = []
    for base in (130, 240):
        rng = np.random.default_rng(20261026 + base)
        t = np.arange(8 * SR) / SR
        phrases = [[.3, 2.1], [2.6, 4.3], [4.8, 6.2], [6.7, 7.7]]
        voiced = in_regions(t, phrases)
        # A short aperiodic interval inside each phrase is a fricative proxy.
        fricatives = [[1.1, 1.22], [3.4, 3.52], [5.5, 5.62], [7.2, 7.32]]
        voiced &= ~in_regions(t, fricatives)
        f0 = base * 2 ** ((2.8 * np.sin(2 * np.pi * .65 * t) +
                          .6 * np.sin(2 * np.pi * 1.7 * t)) / 12)
        phase = 2 * np.pi * np.cumsum(f0) / SR
        signal = np.zeros(len(t))
        for harmonic in range(1, 25):
            # Fixed broad formants; deliberately not a realistic spoken corpus.
            frequency = harmonic * f0
            envelope = (.15 + 2 * np.exp(-.5 * ((frequency - 550) / 150) ** 2)
                        + np.exp(-.5 * ((frequency - 1500) / 220) ** 2)) / harmonic
            signal += envelope * np.sin(harmonic * phase)
        amplitude = np.zeros(len(t))
        for start, end in phrases:
            amplitude += np.clip((t - start) / .025, 0, 1) * np.clip((end - t) / .025, 0, 1)
        clean = signal * voiced * amplitude * (.7 + .3 * np.sin(2 * np.pi * 3 * t) ** 2)
        clean /= max(np.max(np.abs(clean)), 1e-8) / .65
        clean += .035 * rng.normal(size=len(t)) * in_regions(t, fricatives)
        white = rng.normal(size=len(t))
        music = sum(np.sin(2 * np.pi * note * t) for note in (196, 246.94, 293.66))
        music *= .5 + .5 * np.sin(2 * np.pi * 1.3 * t) ** 2
        creak_regions = [[1.5, 1.95], [3.75, 4.15], [5.7, 6.1]]
        creak = clean + .35 * np.sin(phase / 2) * in_regions(t, creak_regions) * voiced
        variants = {
            "clean": clean, "creak": creak,
            "clipped": np.clip(clean * 5, -1, 1),
            "noise20": add_noise(clean, white, voiced, 20),
            "noise10": add_noise(clean, white, voiced, 10),
            "music10": add_noise(clean, music, voiced, 10),
            "agc_proxy": clean * (.3 + 1.4 * np.sin(.7 * t) ** 2),
            "combined": np.clip(add_noise(creak, white, voiced, 10) * 3, -1, 1),
            "aac64": clean,
        }
        grid = np.arange(0, 8, STEP)
        ref = np.interp(grid, t, f0)
        ref[~(in_regions(grid, phrases) & ~in_regions(grid, fricatives))] = np.nan
        guard = (grid >= .1) & (grid < 7.9)
        for start, end in phrases + fricatives:
            guard &= (np.abs(grid - start) > .06) & (np.abs(grid - end) > .06)
        for variant, audio in variants.items():
            file_id = f"synth_{base}_{variant}"
            path = folder / f"{file_id}.wav"
            sf.write(path, audio, SR, subtype="FLOAT")
            if variant == "aac64":
                encoded = folder / f"{file_id}.m4a"
                subprocess.run([str(ffmpeg), "-nostdin", "-v", "error", "-y", "-i", str(path),
                                "-c:a", "aac", "-b:a", "64k", str(encoded)], check=True, timeout=60)
                subprocess.run([str(ffmpeg), "-nostdin", "-v", "error", "-y", "-i", str(encoded),
                                "-ar", str(SR), "-c:a", "pcm_f32le", str(path)], check=True, timeout=60)
            ambiguous = in_regions(grid, creak_regions) if variant in ("creak", "combined") else np.zeros(len(grid), bool)
            # Period doubling changes the physical fundamental. Exclude these
            # frames from primary accuracy, retain the latent reference for audit.
            eligible = guard.copy()
            if ambiguous.any():
                for start, end in creak_regions:
                    eligible &= ~((grid >= start - .08) & (grid <= end + .08))
            reference = folder / f"{file_id}.npz"
            np.savez(reference, times=grid, f0=ref, eligible=eligible, ambiguous=ambiguous)
            cases.append({"id": file_id, "kind": "synthetic", "variant": variant,
                          "path": str(path.resolve()), "reference": str(reference.resolve()),
                          "phrases": phrases, "segments": [[0., 8.]], "base_hz": base})
    return cases
