"""Create report tables directly from saved benchmark results."""
import csv
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "results/summary.csv").open(encoding="utf-8-sig")))
names = list(dict.fromkeys(row["tracker"] for row in rows if row.get("tracker")))


def number(value):
    return "—" if value in (None, "") else f"{float(value):.2f}"


def table(headers, records):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(str, record)) + " |" for record in records])


aggregated = []
for name in names:
    synth = [r for r in rows if r.get("tracker") == name and r.get("kind") == "synthetic" and r["status"] == "ok"]
    def mean(field):
        values = [float(r[field]) for r in synth if r.get(field)]
        return statistics.mean(values) if values else None
    real = [r for r in rows if r.get("tracker") == name and r.get("kind") == "real" and r["status"] == "ok"]
    speed = sum(float(r["runtime_seconds"]) for r in real) / sum(float(r["duration_seconds"]) for r in real) * 60
    aggregated.append([name, *[number(mean(k)) for k in ("gpe_pct", "octave_error_pct", "voicing_error_pct", "voiced_recall_pct", "f0_frame_error_pct")], number(speed)])

block = "### Akurasi sintetis dan runtime lokal\n\n"
block += "Persentase error adalah **rata-rata per file sintetis** (18 file, bukan 18 pembicara). GPE/octave error hanya frame keduanya voiced; recall dan frame error wajib dibaca bersama. Runtime adalah total median runtime empat video dibagi total durasinya, dikali 60.\n\n"
block += table(["Tracker", "GPE %", "Octave error %", "Voicing error %", "Voiced recall %", "F0 frame error %", "Detik/menit video CPU"], aggregated)
block += "\n\n### Dua video bermasalah — tanpa ground truth\n\n"
real_rows = [r for r in rows if r["file_id"] in ("real_02", "real_04")]
block += table(["File", "Tracker", "Median Hz", "Range st", "SD frasa st", "Jump ≥9 st", "Coverage %"],
               [[r["file_id"], r["tracker"], *[number(r.get(k)) for k in ("f0_median_hz", "range_st", "phrase_sd_st")],
                 r["octave_jump_count"], number(r.get("coverage_pct"))] for r in real_rows])
block += "\n\nRange/SD yang mengecil belum membuktikan akurasi; perubahan coverage bisa mengubah statistik.\n"

full = "# Tabel lengkap benchmark\n\n" + block
full += "\n## Seluruh video\n\n"
full += table(["File", "Tracker", "Range st", "SD frasa st", "Jump", "Coverage %"],
              [[r["file_id"], r["tracker"], number(r.get("range_st")), number(r.get("phrase_sd_st")),
                r.get("octave_jump_count", ""), number(r.get("coverage_pct"))] for r in rows if r.get("kind") == "real"])
full += "\n\n## Per kondisi sintetis (rata-rata dua kontur dasar)\n\n"
conditions = []
for variant in dict.fromkeys(r.get("variant") for r in rows if r.get("kind") == "synthetic"):
    for name in names:
        items = [r for r in rows if r.get("variant") == variant and r.get("tracker") == name and r["status"] == "ok"]
        conditions.append([variant, name, *[number(statistics.mean(float(r[k]) for r in items)) for k in
                                          ("gpe_pct", "voicing_error_pct", "voiced_recall_pct", "range_st", "phrase_sd_st")]])
full += table(["Kondisi", "Tracker", "GPE %", "VDE %", "Recall %", "Range st", "SD frasa st"], conditions)
full += "\n\n## Plot setiap file\n\n"
for file_id in dict.fromkeys(r["file_id"] for r in rows):
    full += f"- [{file_id}](plots/{file_id}.png)\n"
(HERE / "results/summary.md").write_text(full, encoding="utf-8")
report = HERE.parent / "riset-pitch-tracking.md"
content = report.read_text(encoding="utf-8")
start, end = "<!-- BENCHMARK_TABLES_START -->", "<!-- BENCHMARK_TABLES_END -->"
content = content.split(start)[0] + start + "\n" + block + "\n" + end + content.split(end)[1]
report.write_text(content, encoding="utf-8")
errors = [r for r in rows if r["status"] != "ok"]
print(json.dumps({"rows": len(rows), "errors": errors, "aggregate": aggregated}, indent=2))
