# Voice Analysis GasGawe — Fase 1

CLI lokal untuk pengukuran prosodi video profil bahasa Indonesia. Python **3.11 64-bit**, Windows, CPU. Konsep tetap mengikuti [fitur-intonasi.md](../fitur-intonasi.md); detail tahapan ada di [rancangan-develop-intonasi.md](../rancangan-develop-intonasi.md).

**Shadow mode selalu aktif.** Hasil bukan ukuran motivasi atau kemampuan kerja dan belum boleh memengaruhi GII. Ambang pitch/QC adalah hipotesis pilot. `null` berarti tidak tersedia/tidak dinilai, bukan nilai nol.


## Halaman tes (paling mudah)

Klik dua kali **`jalankan-tes.bat`** di folder ini. Browser terbuka sendiri ke alamat yang tercetak di jendela hitam (biasanya `http://127.0.0.1:8791/`). Jika port itu dipakai program lain, server otomatis memakai port berikutnya, jadi selalu ikuti alamat yang tercetak.

1. Klik **Pilih video**. Video langsung bisa diputar sementara analisis berjalan (±1× durasi video di CPU; pertama kali lebih lama karena memuat model).
2. Setelah selesai, kanan atas menampilkan keputusan item: **Terpenuhi / Tidak terpenuhi / Perlu peninjauan manual / Bukti belum cukup**, zona nada, zona tempo, dan alasannya.
3. Putar video: grafik nada (CREPE, semitone relatif terhadap nada tengah orang itu) berjalan mengikuti video, garis putus kuning = P10 dan P90 seluruh video (jaraknya = rentang nada), area gelap = jeda. Angka "s.d. kini" dihitung dari awal sampai detik yang diputar. Transkrip menyorot kata yang sedang diucapkan; klik kata atau grafik bawah untuk melompat.

Server hanya mendengarkan di 127.0.0.1 (tidak bisa diakses dari komputer lain). Video ditulis ke folder sementara, dianalisis dengan pipeline yang sama dengan CLI, lalu dihapus. Tutup jendela hitam (atau Ctrl+C) untuk menghentikan server. Port awal lain: `jalankan-tes.bat --port 9000`.

## Instalasi Windows

1. Pasang Python 3.11 x64 dari [python.org](https://www.python.org/downloads/windows/). Pastikan `py -3.11 --version` berfungsi. Jika import paket native gagal karena DLL, pasang Microsoft Visual C++ Redistributable x64 dari Microsoft.
2. Pasang FFmpeg Windows dari tautan build pada [ffmpeg.org](https://ffmpeg.org/download.html#build-windows), misalnya build essentials [Gyan](https://www.gyan.dev/ffmpeg/builds/). Ekstrak lalu tambahkan folder `bin` (berisi **ffmpeg.exe dan ffprobe.exe**) ke PATH. Tidak cukup memasang paket Python bernama ffmpeg.
3. Jalankan PowerShell dari folder `voice-analysis`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchaudio==2.8.0+cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
ffmpeg -version
ffprobe -version
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` memuat pin dependency langsung. `requirements-lock.txt` memuat semua versi lingkungan Windows yang diuji, termasuk dependency transitif. Setuptools dipin karena CTranslate2 4.6 memerlukan `pkg_resources`. Jika menyusun lingkungan baru dari dependency langsung, gunakan `-r requirements.txt`; simpan lock baru dan uji lagi sebelum mengganti versi.

Instalasi membutuhkan internet, disk beberapa GB, dan unduhan bobot ASR pada penggunaan pertama. Analisis audio berjalan lokal; audio tidak diunggah ke Hugging Face. Cache model default `.cache/models`. Silero memakai model ONNX yang sudah ada dalam paket pip. Model ASR default `small` dengan commit yang dipin di `app/config.py`; hash berkas bobot dicatat pada hasil.

Lingkungan kerja sesi pengembangan ini juga memiliki `.venv` dan runtime portabel dalam `.runtime`, tanpa instalasi global. Untuk memakai ffmpeg portabel yang sudah diunduh:

```powershell
$ffmpegBin = Get-ChildItem .runtime\ffmpeg -Filter ffmpeg.exe -Recurse | Select-Object -First 1 -ExpandProperty DirectoryName
$env:PATH = $ffmpegBin + ';' + $env:PATH
.\.venv\Scripts\python.exe --version
```

## Menjalankan

Dengan environment aktif (`.\.venv\Scripts\Activate.ps1`) atau Python yang sesuai:

```powershell
python scripts/analyze_file.py "D:\video\profil.mp4"
python scripts/batch_pilot.py "D:\video\pilot" "hasil-pilot.csv"
python scripts/analyze_file.py "D:\video\profil.mp4" --config config.local.json
python -m pytest -q
```

Aktivasi tidak wajib: ganti `python` dengan `.\.venv\Scripts\python.exe` jika kebijakan PowerShell memblokir Activate.ps1. Dalam sandbox Windows yang tidak bisa mengakses temporary pytest bawaan, gunakan direktori baru khusus tes:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.runtime/pytest-local
```

Pytest mengelola/membersihkan `--basetemp`; jangan arahkan ke folder berisi data pengguna. Audio test dibuat saat runtime. Model Whisper tidak diunduh oleh pytest. Test decoder dapat skip jika ffmpeg tidak ada; test Silero ONNX memakai model paket asli tanpa jaringan.

stdout CLI file berisi JSON UTF-8; log langkah/durasi/error masuk stderr. Pada Windows PowerShell 5, simpan stdout memakai `Out-File -Encoding utf8` bila perlu (default redirection versi tersebut dapat menjadi UTF-16):

```powershell
.\.venv\Scripts\python.exe scripts/analyze_file.py "profil.mp4" 2> analysis.log | Out-File hasil.json -Encoding utf8
```

- Exit **0**: proses selesai, termasuk `bukti_belum_cukup`/`perlu_peninjauan` tanpa kegagalan modul.
- Exit **1**: ada kegagalan modul/IO. JSON parsial tetap dicetak. Batch tetap memproses video berikutnya.
- Exit **2**: argumen/config/folder batch invalid; pesan di stderr.
- Batch memproses folder **nonrekursif**, urutan nama deterministik; ekstensi didukung ada di config. Satu baris per video termasuk video rusak. Output CSV yang sudah ada ditimpa. Nilai null menjadi sel kosong; daftar/objek kompleks menjadi JSON dalam sel. Kolom bertingkat memakai titik.

CSV/JSON mengandung nama file dan transkrip. Lindungi akses/retensinya; jangan masukkan hasil pribadi, cache atau media ke git. Log tidak memuat transkrip.

## Konfigurasi

Semua ambang/parameter dalam dataclass `Config` di `app/config.py`. `--config` menerima objek JSON dengan sebagian field uppercase untuk override; nama/tipe yang salah ditolak. Contoh:

```json
{
  "MODE": "ekspresivitas",
  "ASR_CPU_THREADS": 4,
  "ASR_LOCAL_FILES_ONLY": true,
  "FFMPEG": "C:/tools/ffmpeg/bin/ffmpeg.exe",
  "FFPROBE": "C:/tools/ffmpeg/bin/ffprobe.exe",
  "PAUSE_MIN_SECONDS": 0.25
}
```

- Default `ASR_LOCAL_FILES_ONLY=false` memungkinkan unduh bobot. Aktifkan true setelah bobot tersedia. Gunakan `ASR_LOCAL_PATH` untuk folder model CTranslate2 lokal; hash berkas tetap dicatat.
- Mengganti ukuran model membutuhkan **ASR_MODEL_SIZE, ASR_REPO dan ASR_REVISION** yang cocok. Revision remote harus commit SHA 40 karakter, bukan `main`. `ASR_LOCAL_PATH` melewati pemeriksaan repo/revision; isi model menjadi tanggung jawab pengguna dan diaudit lewat hash.
- `ASR_ENABLED=false` berguna untuk debug akustik. Transkripsi/tempo null, alasan `asr_disabled`, keputusan biner null.
- **Default `MODE="ekspresivitas"` (disetujui 27-10-2026):** item intonasi terpenuhi jika nada di **zona wajar** (rentang ≥4 st dan SD frasa ≥2 st, tetapi tidak di atas 15 st / 7 st) **dan** tempo tidak lambat (articulation rate ≥4,5 suku kata/detik). Nada di atas batas atas → `pitch_above_plausible_zone` (biasanya salah ukur) → peninjauan manual, tidak otomatis lulus. Tempo dalam ±0,25 dari ambang → `tempo_near_pilot_threshold` → peninjauan manual. Tempo tidak tersedia (ASR/silabifikasi gagal) → `tempo_unavailable`. Output menambah `pitch_zone` (datar/wajar/ekstrem) dan `tempo_zone` (lambat/wajar/cepat). `MODE="intonasi"` = nada saja, tempo hanya deskripsi. Semua ambang hipotesis pilot dari 4 video sampel. Bicara cepat (>9 suku kata/detik) hanya deskripsi, tidak menggagalkan item.
- `ANALYSIS_VERSION="0.3.0"` harus berubah ketika semantik ekstraksi/aturan berubah. Simpan snapshot config dan hash untuk tiap eksperimen.
- Default batas: input 512 MiB, durasi 600 detik, PCM 60 juta sampel lintas channel, timeout tiap proses ffmpeg/ffprobe 120 detik. File <1 detik ditolak jelas; speech terdeteksi <30 detik menghasilkan bukti belum cukup.
- Hipotesis pitch: P90–P10 ≥4 st dan median SD per proksi frasa ≥2 st, coverage ≥0,3. Zona dekat ambang ±0,25 st (range) / ±0,15 st (SD) memicu review. Parameter ini belum norma Indonesia.

## Definisi dan alur

1. `ffprobe` memilih stream audio pertama. Decode pada sample rate/channel asli; clipping dihitung pada PCM ini. Korelasi stereo anti-phase / satu channel sangat lemah memicu pemilihan channel RMS lebih tinggi dan review. >2 channel juga memakai pilihan RMS dan review. Heuristik ini tidak mengidentifikasi speaker yang benar.
2. Salinan mono float diresample ke 16 kHz, tanpa denoising atau normalisasi. Berkas decode hanya temporer dan dibersihkan setelah proses.
3. Silero ONNX mendeteksi speech tanpa padding. Jeda internal ≥250 ms dikeluarkan dari durasi artikulasi; jeda <250 ms tetap di dalamnya. Awal/akhir diam di luar tuturan tidak masuk penyebut tempo.
4. QC akhir mengestimasi SNR dari frame speech penuh dan non-speech dengan margin 100 ms. `10 log10((P_speech+noise - P_noise)/P_noise)`. Noise nonstasioner, musik, dan kesalahan VAD dapat merusak estimasi; tanpa referensi memadai hasil null.
5. Pitch utama: **CREPE tiny** (torchcrepe, MIT, decoder Viterbi) 50–600 Hz pada audio speech-only. Frame dianggap voiced jika periodicity ≥0,21 **dan** energi frame ≥−40 dB relatif terhadap P90 energi bicara orang itu sendiri (tidak bergantung volume rekaman; gate Silence bawaan torchcrepe bersifat absolut sehingga tidak dipakai). Praat raw autocorrelation (two-pass, kuartil/1,5) tetap dihitung sebagai `pitch.audit_praat`, **tidak** memengaruhi skor. Alasan pergantian: pada rekaman HP nyata Praat membaca dengung/bising latar 50–100 Hz sebagai nada (range ±25 st, ±20 octave jump), lihat `../research/riset-pitch-tracking.md` dan `../research/hasil-crepe/ringkasan.md`. `PITCH_TRACKER="praat"` mengembalikan perilaku lama. Konversi `12*log2(F0/median(F0_valid))`; tidak membagi dengan SD pribadi. CREPE terkuantisasi 20 sen + dither (seed tetap), sehingga nada datar tetap tersebar ±0,2 st.
6. ASR mengolah rentang speech pertama–terakhir (termasuk jeda), CPU int8, bahasa `id`, timestamp kata. Tidak memakai VAD filter ASR sebagai definisi durasi. Timestamp kata hanya audit, bukan batas artikulasi.
7. Silabifikasi menormalisasi angka/singkatan; menghitung inti vokal dengan diftong akhir ai/au/oi/ei dan pengecualian hiatus. Filler/repetisi yang muncul di transkrip tetap dihitung. Angka tahun/telepon, akronim, nama, dan campur kode tetap bisa berbeda pelafalannya dari asumsi.
8. Scoring hanya memakai pitch yang dapat diukur; tempo/jeda adalah deskripsi numerik. Modul gagal → fitur terkait null + alasan abstain. SNR rendah sendirian tidak memaksa abstain. Clipping berat, coverage rendah, octave jump berlebih, masalah channel, atau dekat ambang → review. Ekstraksi independen yang masih bisa berjalan dilanjutkan.

**Konvensi pilot suku kata:** `mempertanggungjawabkan` di-override menjadi **8**, memenuhi acceptance yang diminta. Pemenggalan `mem-per-tang-gung-ja-wab-kan` berjumlah 7; kode memberi flag `pilot_override_mempertanggungjawabkan_8`. Ini perlu diselesaikan sebelum validasi tempo. `pantai=2`, `saya=2`. Tidak mengklaim rule-based sebagai silabifikasi fonetik sempurna.

## Field output

Semua timestamp dalam detik dari **awal audio asli**, interval `[start,end)`. Semua angka bisa null bila tak terukur, kecuali yang memang dihitung sebagai nol. Core schema mengikuti konsep §9.1; penambahan field bersifat audit.

| Field | Arti / satuan |
|---|---|
| `analysis_version` | Versi semantik analisis. |
| `mode`, `shadow_mode` | Konstruk dipilih; shadow selalu true pada Fase 1. |
| `input_file` | Path input sebagaimana diberikan. |
| `status` | `dapat_dinilai`, `bukti_belum_cukup`, atau `perlu_peninjauan`. |
| `abstain_reason` | Kode alasan dipisah `;`; null hanya jika ada keputusan. |
| `intonation_item_passed` | Boolean hipotesis pilot jika dapat dinilai; selain itu null. Bukan keputusan rekrutmen. |
| `descriptors` | Kalimat Indonesia netral berdasarkan angka dan status. |
| `valid_speech_seconds` | Jumlah durasi speech terdeteksi VAD tanpa padding, bukan sertifikat bebas kontaminasi. |
| `speech_span_seconds` | Waktu speech pertama sampai terakhir; termasuk jeda internal. |
| `articulation_seconds` | Rentang tuturan dikurangi jeda internal ≥batas config; konsonan tanpa F0 tidak dibuang. |
| `f0_valid_coverage` | Frame F0 valid dibagi frame analisis dalam speech; rasio 0–1, bukan persen seluruh video. |
| `f0_valid_seconds` | Jumlah frame F0 valid × langkah analisis; estimasi durasi, bukan jumlah segmen. |
| `f0_median_hz` | Median F0 valid sebagai referensi semitone. Tidak dipakai menebak gender. |
| `pitch_range_p10_p90_st` | P90−P10 semitone seluruh frame speech voiced valid. |
| `pitch_sd_st_phrase_median` | Median SD populasi (ddof=0) dalam setiap proksi frasa; tiap frasa berbobot sama. |
| `octave_jump_count`, `octave_jump_per_min` | Lonjakan ≥9 st antarframe voiced bertetangga dalam segmen sama; rate per menit speech VAD. |
| `speech_rate_syl_per_s`, `articulation_rate_syl_per_s` | Jumlah suku kata / rentang tuturan, atau / durasi artikulasi. |
| `speech_rate_wpm`, `articulation_rate_wpm` | Jumlah kata setelah normalisasi ×60 / penyebut yang sama. |
| `word_count`, `syllable_count` | Hitungan estimasi dari transkrip yang dinormalisasi, bukan token subword Whisper. |
| `pause_ratio` | Total jeda internal ≥batas / rentang tuturan. |
| `pause_median_seconds`, `pause_p90_seconds` | Median/P90 jeda terpilih. Null jika tidak ada jeda. |
| `long_pause_per_min` | Jumlah jeda >2 detik per menit rentang tuturan. Nol jika ada speech tanpa jeda panjang. |
| `energy_end_vs_start_db` | Selalu null: modul energi belum dibuat pada Fase 1. |
| `evidence_timestamps` | Maksimal tiga interval proksi frasa dengan SD paling dekat median, bukan word-level emphasis. |
| `audio_quality.status` | `ok`/`warning`/`insufficient`; null jika QC akhir gagal/tidak berjalan. |
| `audio_quality.snr_est_db` | Estimasi SNR dalam dB; null jika tidak terdefinisi. |
| `audio_quality.snr_unavailable_reason` | Kode sebab SNR tidak tersedia; null jika tersedia. |
| `audio_quality.clipping_pct` | Persen sampel dengan abs amplitude ≥0,999 pada channel asli terburuk, sebelum downmix/resampling. Bukan rasio 0–1. |
| `audio_quality.valid_speech_seconds` | Salinan hasil durasi VAD untuk audit QC. |
| `audio_quality.warnings` | Kode peringatan QC; SNR tidak tersedia juga di sini. |
| `audio_info` | Metadata dan pemeriksaan audio asli; null jika ekstraksi gagal. |
| `audio_info.original_channels`, `original_sample_rate`, `original_channel_layout`, `original_codec` | Channel, Hz, layout, codec input stream pertama. |
| `audio_info.duration_seconds`, `analysis_sample_rate` | Durasi decode dan sample rate salinan analisis (16.000 Hz). |
| `audio_info.rms_per_channel`, `peak_per_channel`, `clipping_pct_per_channel` | RMS/peak PCM relatif full scale serta clipping persen tiap channel. |
| `audio_info.stereo_correlation` | Korelasi L/R; null untuk non-stereo atau varians nol. |
| `audio_info.selected_channel`, `channel_strategy` | Indeks channel mulai 0 bila dipilih; null berarti rata-rata channel. Strategy `select` atau `mean`. |
| `audio_info.clipping_pct`, `warnings`, `ffmpeg_version` | Nilai clipping maksimum, warning pemilihan channel, versi executable. |
| `vad.segments`, `vad.phrases`, `vad.pauses` | Speech asli; proksi frasa setelah penggabungan celah pendek; jeda internal terpilih. |
| `vad.model_id`, `vad.model_sha256` | Identitas dan hash model ONNX bawaan Silero. Metrik VAD lain mengulang top-level. |
| `pitch.phrase_statistics` | Per frasa: start/end, valid_voiced_seconds, sd_st, range_st. Frasa dengan <0,3 detik F0 valid tidak dipakai. |
| `pitch.tracking` | Tracker yang dipakai (`torchcrepe_tiny`/`praat_ac`), parameter gating atau batas Hz pass1/pass2, speech_frame_count, valid_frame_count, step_seconds, versi. Metrik pitch lain mengulang top-level. |
| `pitch.audit_praat` | Range, SD frasa, median, coverage, octave jump versi Praat untuk audit perbedaan tracker. Tidak dipakai scoring. |
| `transcription.text`, `words`, `language`, `span_start`, `span_end` | Teks ASR, daftar kata, bahasa dan cakupan ASR. Tiap kata berisi word/start/end/probability; probability bukan keandalan yang dikalibrasi. |
| `transcription.model` | repo/revision/source, file_sha256, device, compute_type, size; hash juga tersedia untuk model lokal. |
| `syllables` | normalized_text, word_count, syllable_count, rule_version dan warnings normalisasi. |
| `errors` | Daftar module/code/message/exception_type. Kegagalan satu modul tidak membuang fitur lain. |
| `warnings` | Gabungan kode peringatan channel, QC, normalisasi, atau ASR dinonaktifkan. |
| `module_status` | `ok`/`failed`/`skipped` untuk extract, vad, quality, pitch, transcribe, syllable, tempo. |
| `timings_seconds` | Waktu tiap modul dan total; cold start/download ikut dihitung. |
| `provenance` | Versi python/platform/packages, snapshot config dan config_sha256. Hash model/ffmpeg ada dalam hasil modul. |
| `limitations` | Batas interpretasi dan fitur yang belum tersedia. |

## Keterbatasan dan tahap berikutnya

- Belum ada kalibrasi atau evaluasi video pelamar nyata, logat, disabilitas bicara, maupun perangkat. Tes sinus hanya memeriksa implementasi F0.
- Ambang, SNR, clipping, adaptive pitch dan VAD adalah heuristik; octave jump tidak otomatis dikoreksi. Codec lossy dapat menyamarkan clipping yang sudah terjadi.
- AR mengandalkan VAD sebagai pendekatan durasi dan jumlah suku kata ASR; belum ada forced alignment, evaluasi WER/filler recall atau detektor inti suku kata akustik.
- Whisper bisa menghasilkan teks pada hening: smoke test langsung pada 3 detik nol menghasilkan "Terima kasih.". Pipeline melewati ASR bila VAD tidak mendeteksi speech; ini tidak menjamin bebas halusinasi pada jeda internal atau salah deteksi VAD. Jangan menganggap timestamp/probabilitas kata sebagai validasi kebenaran.
- Model CPU tetap bisa lambat. Batas durasi/thread dan timeout ffmpeg tersedia; belum ada hard timeout/cancel untuk native ASR (memerlukan worker proses). Batch berurutan, model dicache dalam proses; jangan memanggil model Silero stateful ini secara paralel tanpa isolasi.
- Belum ada deteksi musik/overlap/gema/dropout, energi relatif, penekanan lokal, SER, API, queue, Docker, database, atau integrasi GII.
- Pin dan hash membantu reproduksi; perbedaan CPU/library masih dapat menyebabkan selisih numerik kecil. Audit lisensi sebelum distribusi/produksi, termasuk GPL Parselmouth/Praat dan build FFmpeg yang dipilih.
- Fase 2: audit contoh manual, stabilkan rubrik, uji error per kelompok serta 200/300 ms, dan kumpulkan data cukup sebelum menilai kelayakan produksi.

Referensi implementasi: [torchcrepe](https://github.com/maxrmorrison/torchcrepe), [Parselmouth API](https://parselmouth.readthedocs.io/en/stable/api_reference.html), [Silero VAD](https://github.com/snakers4/silero-vad), [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
