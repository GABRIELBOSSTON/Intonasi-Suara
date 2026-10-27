"""Record local binary/model provenance and protect application sources."""
import importlib.metadata
import json
from pathlib import Path
import sys
import winreg

sys.dont_write_bytecode = True
from run_benchmark import HERE, ROOT, sha, dump
import numpy as np
import parselmouth
from parselmouth.praat import call
import torchcrepe
import swift_f0

evidence = {"parselmouth": parselmouth.__version__, "praat": parselmouth.PRAAT_VERSION}
try:
    call(parselmouth.Sound(np.sin(np.arange(16000)), sampling_frequency=16000),
         "To Pitch (filtered autocorrelation)", .01, 50, 800, 15, False, .03, .09, .5, .055, .35, .14)
    evidence["filtered_ac_available"] = True
except Exception as exc:
    evidence["filtered_ac_available"] = False
    evidence["filtered_ac_error"] = str(exc)
with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
    evidence["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
evidence["torchcrepe_parameters"] = {size: sum(p.numel() for p in torchcrepe.Crepe(size).parameters()) for size in ("tiny", "full")}
evidence["models"] = []
for module, pattern in ((torchcrepe, "*.pth"), (swift_f0, "*.onnx")):
    for path in Path(module.__file__).parent.rglob(pattern):
        evidence["models"].append({"package": module.__name__, "file": path.name,
                                   "bytes": path.stat().st_size, "sha256": sha(path)})
before = json.loads((HERE / "app-hashes-before.json").read_text(encoding="utf-8-sig"))
evidence["app_changed_since_research_start"] = [item["path"] for item in before if sha(item["path"]).lower() != item["sha256"].lower()]
evidence["packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
dump(HERE / "environment-audit.json", evidence)
print(json.dumps({key: value for key, value in evidence.items() if key != "packages"}, indent=2))
