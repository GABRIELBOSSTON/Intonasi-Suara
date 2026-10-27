# Benchmark pitch lokal — GasGawe

Eksperimen terpisah; tidak mengubah `voice-analysis/app/`, tidak memakai SER, tidak mengunggah audio. Video dibaca lokal, PCM temporer dihapus sesudah decode. Kontur dan plot tetap merupakan hasil turunan data pelamar: simpan lokal dan ikuti retensi yang sama. Fixture sintetis bebas dari data pelamar.

## Jalankan di Windows / Python 3.11

Dari root `D:\gasgawe\prototipe\notasi`:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD 'voice-analysis\.runtime\uv-cache'
& voice-analysis/.runtime/uv/uv.exe pip install --python voice-analysis/.venv/Scripts/python.exe -c voice-analysis/requirements-lock.txt -r research/pitch-benchmark/requirements.txt
$env:PYTHONDONTWRITEBYTECODE = '1'
& voice-analysis/.venv/Scripts/python.exe -m pytest research/pitch-benchmark/test_benchmark.py -q
& voice-analysis/.venv/Scripts/python.exe research/pitch-benchmark/run_benchmark.py
& voice-analysis/.venv/Scripts/python.exe research/pitch-benchmark/audit_environment.py
```

Model neural disertakan dalam paket terpasang; inferensi dan VAD tidak memerlukan jaringan. ffmpeg/ffprobe dicari di `voice-analysis/.runtime/ffmpeg/`. Runtime CPU dibatasi empat thread untuk Torch/ONNX dan variabel OpenMP/MKL, bukan isolasi seluruh mesin. Default dua pengulangan setelah warmup; median waktu dilaporkan. pYIN pertama kali dapat memerlukan kompilasi Numba. `requirements-lock.txt` mencatat seluruh environment sesudah pemasangan; versi Torch CPU memakai indeks PyTorch dari petunjuk instalasi aplikasi jika membangun environment baru. Jangan memakai lock ini untuk mengganti environment aplikasi tanpa meninjau perubahan.

Opsi:

```powershell
# Hanya empat video, atau satu fixture untuk smoke test
& voice-analysis/.venv/Scripts/python.exe research/pitch-benchmark/run_benchmark.py --only real_ --output research/pitch-benchmark/real-repeat
& voice-analysis/.venv/Scripts/python.exe research/pitch-benchmark/run_benchmark.py --only synth_130_clean --repeats 1 --output research/pitch-benchmark/smoke-repeat
# CREPE full opsional; jauh lebih berat pada CPU yang diuji
& voice-analysis/.venv/Scripts/python.exe research/pitch-benchmark/run_benchmark.py --only synth_130_clean --trackers torchcrepe_full --output research/pitch-benchmark/full-repeat
```

`--samples <folder>`, `--trackers <nama...>`, `--resume` tersedia. Resume memeriksa hash input, config, skrip benchmark, versi paket, daftar tracker dan jumlah pengulangan. Skrip menulis ulang hasil pada folder output yang sama; gunakan folder baru untuk eksperimen berbeda. Kegagalan per tracker/file tercatat di CSV dan proses dilanjutkan; periksa kolom `status`.

## Data dan perbandingan

18 fixture: dua dasar F0 (130/240 Hz), masing-masing clean, subharmonik/creak, hard clipping, white noise 20/10 dB, chord musik 10 dB, AGC-proxy, gabungan creak/noise/clipping, AAC 64 kbps. Durasi masing-masing 8 detik. Ini sinyal harmonik dengan formant lebar dan fricative-proxy, **bukan rekaman manusia/TTS realistis**. Tidak menyimulasikan reverb, multi-speaker atau semua fonasi manusia. SNR noise/musik dihitung dari daya pada bagian target bersuara sebelum pencampuran. AGC-proxy hanya gain pelan yang berubah, bukan implementasi AGC HP.

Praat current meniru app yang terpasang; tes memverifikasi kesamaan statistik. Hirst hanya mengganti floor menjadi `0.75*P25`, ceiling tetap `1.5*P75`, dan batas global 50–600 Hz dipertahankan untuk perbandingan terkontrol (bukan replika persis pass pertama 60–750 Hz paper). `praat_hirst_cost04` menambahkan octave cost 0.04 **sebagai ablation heuristik**, bukan angka rekomendasi universal paper. Postprocessing membuang deviasi >8 st dari median lokal 11 frame jika ada ≥5 tetangga valid; tidak menggandakan/membagi F0 atau mengisi gap. Ini sengaja konservatif dan tidak mengatasi salah oktaf panjang.

pYIN memakai 16 kHz, frame 1024, hop 160, 50–600 Hz, `fill_na=NaN`. CREPE tiny memakai decoder Viterbi, periodicity median 3 frame, threshold 0.21, silence gate −60 dB. Ambang ini berasal dari contoh repo, belum dikalibrasi untuk Indonesia. Seed NumPy/Torch 20261026 per pemanggilan mengunci dither internal torchcrepe. SwiftF0 0.3.0 memakai confidence ≥0.5 dan rentang 50–600 Hz, hop native 16 ms.

Pada video, decoder dan Silero VAD aplikasi diimpor baca-saja untuk menjaga preprocessing sama. Praat memakai audio dengan non-speech dimask, persis baseline aplikasi; neural/pYIN menerima sinyal penuh. Statistik seluruh tracker dibatasi mask VAD yang sama. Dengan demikian perbandingan mencakup konfigurasi pipeline praktis, bukan isolasi algoritme murni. Pada sintetis semua tracker menerima sinyal penuh; tidak ada label ground truth yang dipakai untuk keputusan voiced atau F0. Frasa ground truth hanya dipakai menghitung statistik per frasa.

Range/coverage global sintetis memakai seluruh jendela audio, termasuk sela/fricative; pitch palsu pada musik di sela ikut masuk. Range/coverage global video memakai speech VAD. Denominator keduanya berbeda dan tidak untuk perbandingan lintas jenis data. Ground truth accuracy sintetis tetap memeriksa voiced maupun unvoiced.

## Definisi metrik / field

- `gpe_pct`: persentase `abs(F0_est/F0_ref − 1)>0.20` pada frame yang **keduanya voiced**. Null bila tidak ada frame bersama.
- `octave_error_pct`: error absolut 12±1 st, denominator frame keduanya voiced. Tidak menghitung semua kelipatan oktaf.
- `voicing_error_pct`: voiced/unvoiced tidak cocok ÷ seluruh frame eligible.
- `voiced_recall_pct`: frame keduanya voiced ÷ frame referensi voiced. `false_alarm_pct`: estimasi voiced pada referensi unvoiced ÷ frame referensi unvoiced.
- `f0_frame_error_pct`: (kesalahan voicing + GPE pada frame bersama) ÷ seluruh frame eligible. Membatasi kesan akurasi bagus akibat membuang hampir semua F0.
- `median_abs_error_st`: median selisih absolut semitone pada frame voiced bersama.
- `eligible_frames`, `joint_voiced_frames`: denominator untuk audit. Grid evaluasi 10 ms, nearest frame dengan toleransi setengah hop; tidak interpolasi melewati unvoiced.
- Batas phonation/fricative diberi guard 60 ms, ujung audio 100 ms. Subharmonik creak mengubah periodisitas fisik: frame ambigu dan margin 80 ms dikeluarkan dari akurasi utama. `ambiguous_latent_gpe_pct` hanya diagnostik terhadap F0 laten, **bukan bukti kesalahan fisiologis tracker**.
- `f0_median_hz`, `range_st`: median F0 dan P90−P10 semitone pada voiced dalam speech. Semitone = `12 log2(F0/reference)`; referensi konstan tidak memengaruhi SD/range.
- `phrase_sd_st`, `phrase_range_st`: median SD populasi/range per frasa dengan ≥0.3 detik voiced. `mad_st`: median absolute deviation F0 dalam semitone.
- `coverage_pct`: voiced valid ÷ frame dalam speech. VAD speech mencakup konsonan unvoiced sehingga 100% bukan target.
- `octave_jump_count`: selisih ≥9 st antara frame valid bersebelahan dalam segmen VAD yang sama. Ini indikator diskontinuitas, **bukan octave error terhadap ground truth**. `octave_jump_per_min` memakai jumlah durasi segmen speech sebagai denominator.
- `runtime_seconds`: median durasi pemanggilan tracker setelah warmup; `seconds_per_audio_minute` = runtime/durasi audio ×60. `runtime_min_seconds/max_seconds` menyimpan dua pengukuran; tidak termasuk decode, VAD, plot atau startup/import/model loading.
- `floor/ceiling/octave_cost/rejected_pct`: metadata adaptor Praat; `periodicity_threshold/confidence_threshold` metadata neural.
- `kind=real`: semua metrik akurasi ground truth kosong. Kontur yang tampak lebih rapi belum membuktikan lebih benar.

## Artefak

- `results/summary.csv`: satu baris per file×tracker; `status/error` menyimpan kegagalan.
- `results/contours/*.csv`: F0 tiap tracker pada grid yang sama, kosong untuk unvoiced.
- `results/plots/*.png`: waveform dan kontur; referensi sintetis hitam, daerah creak ambigu oranye.
- `results/cases/*.json`: metrik, metadata channel/decode, VAD, hash input; alias real_01 dst dipetakan ke path asli di sini/manifest.
- `results/manifest.json`: config lengkap, versi paket, hash model dan sumber, waktu warmup, pemeriksaan app tidak berubah.
- `environment-audit.json`: CPU, ukuran/model hash, hitungan parameter, uji ketersediaan filtered AC, integritas app sejak awal riset.
- `smoke/`: uji awal CREPE full; hasil dither smoke awal belum di-seed dan tidak digabung ke tabel akurasi benchmark utama.
- `benchmark.log` dan `pytest-results.txt`: keluaran eksekusi.
- `summarize_results.py`: menghasilkan `results/summary.md` (tabel lengkap dan indeks 22 plot) serta memperbarui blok tabel laporan langsung dari CSV.
- `app-pytest-results.txt`, `decoder-pytest-results.txt`: 213 tes aplikasi lulus, lalu tiga tes ffmpeg yang awalnya skip lulus setelah PATH diperbaiki. Satu deprecation warning dependency Silero tersisa.

Lisensi paket, batas metodologi dan interpretasi ada di [laporan](../riset-pitch-tracking.md). Tidak ada keputusan kelulusan pelamar yang dihasilkan benchmark ini.
