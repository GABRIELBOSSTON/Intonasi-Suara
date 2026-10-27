"""Run a local, resumable benchmark without changing application sources."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "voice-analysis"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache" / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / ".cache" / "numba"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import soundfile as sf
from app.config import Config
from app.pipeline.extract import extract_audio
from app.pipeline.vad import analyze_vad
from generate import make_cases, SR
from metrics import align, accuracy, summarize
from trackers import NAMES, DEFAULT_NAMES, Trackers


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_case(case, audio, contours, reference, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(contours) + 1, 1, figsize=(15, 2.0 * (len(contours) + 1)), sharex=True)
    times = np.arange(len(audio)) / SR
    axes[0].plot(times[::16], audio[::16], linewidth=.35, color="gray")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(case["id"] + (" | NO ground truth" if reference is None else " | synthetic known F0"))
    for axis, (name, (t, f0)) in zip(axes[1:], contours.items()):
        axis.plot(t, f0, ".", markersize=1.5, label=name)
        if reference is not None:
            axis.plot(reference["times"], reference["f0"], color="black", alpha=.5, linewidth=.8, label="latent reference")
            ambiguous = reference["ambiguous"]
            axis.fill_between(reference["times"], 45, 650, where=ambiguous, color="orange", alpha=.15)
        axis.set_ylim(45, 650)
        axis.set_yscale("log", base=2)
        axis.set_yticks([50, 100, 200, 400, 600], labels=[50, 100, 200, 400, 600])
        axis.set_ylabel("F0 (Hz)")
        axis.legend(loc="upper right", fontsize=8)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("Time (seconds)")
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=ROOT / "sampel")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--only", help="Substring of case ID for a smoke test")
    parser.add_argument("--trackers", nargs="+", choices=NAMES, default=DEFAULT_NAMES)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    import torch
    import parselmouth
    torch.set_num_threads(4)
    ffmpeg = next((ROOT / "voice-analysis/.runtime/ffmpeg").rglob("ffmpeg.exe"), None)
    if ffmpeg is None:
        parser.error("Portable ffmpeg.exe was not found")
    cfg = Config(FFMPEG=str(ffmpeg), FFPROBE=str(ffmpeg.with_name("ffprobe.exe")))
    args.output.mkdir(parents=True, exist_ok=True)
    for sub in ("plots", "contours", "cases"):
        (args.output / sub).mkdir(exist_ok=True)
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "voice-analysis/app").rglob("*.py")}
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    benchmark_hashes = {p.name: sha(p) for p in HERE.glob("*.py")}
    protocol_hash = hashlib.sha256(json.dumps({"sources": benchmark_hashes, "packages": packages}, sort_keys=True).encode()).hexdigest()
    manifest = {"created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                "threads": 4, "repeats": args.repeats, "praat_version": parselmouth.PRAAT_VERSION,
                "config": cfg.snapshot(), "config_fingerprint": cfg.fingerprint(),
                "packages": packages, "app_hashes_before": source_hashes, "models": [],
                "benchmark_hashes": benchmark_hashes, "protocol_hash": protocol_hash,
                "numpy_torch_seed_per_neural_call": 20261026,
                "protocol": {"grid_seconds": .01, "gpe_relative_threshold": .2,
                             "octave_tolerance_st": 1, "boundary_guard_seconds": .06,
                             "creak_ambiguous_excluded": True, "rtf_excludes_decode_vad_plot": True,
                             "real_vad_used_for_praat_and_statistics": True}}
    cases = make_cases(HERE / "fixtures", ffmpeg)
    if args.samples.is_dir():
        paths = sorted(p for p in args.samples.rglob("*") if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".wav", ".m4a"))
        for i, path in enumerate(paths, 1):
            cases.append({"id": f"real_{i:02d}", "kind": "real", "variant": "original",
                          "path": str(path.resolve()), "source_sha256": sha(path)})
    if args.only:
        cases = [case for case in cases if args.only in case["id"]]
    manifest["cases"] = cases
    dump(args.output / "manifest.json", manifest)
    trackers = Trackers(cfg)
    warm_audio = (.2 * np.sin(2 * np.pi * 180 * np.arange(SR) / SR)).astype(np.float32)
    manifest["warmup_seconds"] = {}
    for name in args.trackers:
        started = time.perf_counter()
        try:
            trackers.run(name, warm_audio, [[0., 1.]])
            manifest["warmup_seconds"][name] = time.perf_counter() - started
            print(f"Warmup OK: {name}", flush=True)
        except Exception as exc:
            manifest["warmup_seconds"][name] = {"error": str(exc)}
            print(f"Warmup FAILED: {name}: {exc}", flush=True)
    all_rows = []
    for case in cases:
        result_path = args.output / "cases" / f"{case['id']}.json"
        if args.resume and result_path.exists():
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if (cached.get("config_fingerprint") == cfg.fingerprint()
                    and cached.get("protocol_hash") == protocol_hash
                    and cached.get("source_sha256") == sha(case["path"])
                    and cached.get("trackers") == args.trackers
                    and cached.get("repeats") == args.repeats):
                all_rows.extend(cached["rows"])
                print(f"Resume: {case['id']}", flush=True)
                continue
        print(f"Case: {case['id']}", flush=True)
        try:
            if case["kind"] == "real":
                with tempfile.TemporaryDirectory(prefix="pitch-", dir=args.output) as tmp:
                    decoded = extract_audio(case["path"], tmp, cfg)
                    audio = decoded["audio"]
                    case["audio_info"] = decoded["audio_info"]
                vad = analyze_vad(audio, SR, cfg)
                case["segments"], case["phrases"] = vad["segments"], vad["phrases"]
                case["vad"] = vad
            else:
                audio, sr = sf.read(case["path"], dtype="float32")
                assert sr == SR
            duration = len(audio) / SR
            reference = dict(np.load(case["reference"])) if case.get("reference") else None
            grid = reference["times"] if reference is not None else np.arange(0, duration, .01)
            contours, rows, aligned = {}, [], {}
            for name in args.trackers:
                row = {"file_id": case["id"], "kind": case["kind"], "variant": case["variant"],
                       "tracker": name, "duration_seconds": duration, "status": "ok"}
                try:
                    runtimes = []
                    for _ in range(args.repeats):
                        started = time.perf_counter()
                        times, f0, meta = trackers.run(name, audio, case["segments"])
                        runtimes.append(time.perf_counter() - started)
                    contours[name] = (times, f0)
                    aligned[name] = align(times, f0, grid)
                    row.update(summarize(times, f0, case["segments"], case["phrases"]))
                    if reference is not None:
                        row.update(accuracy(aligned[name], reference["f0"], reference["eligible"]))
                        amb = reference["ambiguous"]
                        if amb.any():
                            diagnostic = accuracy(aligned[name], reference["f0"], amb)
                            row["ambiguous_latent_gpe_pct"] = diagnostic["gpe_pct"]
                    row.update({"runtime_seconds": float(np.median(runtimes)),
                                "seconds_per_audio_minute": float(np.median(runtimes) / duration * 60),
                                "runtime_min_seconds": min(runtimes), "runtime_max_seconds": max(runtimes),
                                **meta})
                    print(f"  {name}: range={row['range_st']} runtime={row['runtime_seconds']:.3f}s", flush=True)
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    print(traceback.format_exc(), flush=True)
                rows.append(row)
            contour_rows = []
            for i, value in enumerate(grid):
                item = {"time_seconds": value}
                for name, values in aligned.items():
                    item[name] = float(values[i]) if np.isfinite(values[i]) else None
                if reference is not None:
                    item.update(reference_hz=float(reference["f0"][i]) if np.isfinite(reference["f0"][i]) else None,
                                eligible=bool(reference["eligible"][i]), ambiguous=bool(reference["ambiguous"][i]))
                contour_rows.append(item)
            write_csv(args.output / "contours" / f"{case['id']}.csv", contour_rows)
            plot_case(case, audio, contours, reference, args.output / "plots" / f"{case['id']}.png")
            dump(result_path, {"case": case, "rows": rows, "trackers": args.trackers,
                               "protocol_hash": protocol_hash,
                               "config_fingerprint": cfg.fingerprint(), "repeats": args.repeats,
                               "source_sha256": sha(case["path"])})
            all_rows.extend(rows)
        except Exception as exc:
            all_rows.append({"file_id": case["id"], "status": "error", "error": str(exc)})
            print(traceback.format_exc(), flush=True)
        write_csv(args.output / "summary.csv", all_rows)
    for package, patterns in (("torchcrepe", ("*.pth",)), ("swift_f0", ("*.onnx",))):
        module = __import__(package)
        for pattern in patterns:
            for path in Path(module.__file__).parent.rglob(pattern):
                manifest["models"].append({"package": package, "file": path.name,
                                           "bytes": path.stat().st_size, "sha256": sha(path)})
    manifest["app_hashes_after"] = {p: sha(ROOT / p) for p in source_hashes}
    manifest["app_sources_unchanged"] = manifest["app_hashes_after"] == source_hashes
    manifest["finished_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump(args.output / "manifest.json", manifest)
    print(f"Finished {len(cases)} cases; app unchanged: {manifest['app_sources_unchanged']}", flush=True)


if __name__ == "__main__":
    main()
