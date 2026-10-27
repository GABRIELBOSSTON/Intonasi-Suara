"""Prepare local audio and offline Whisper word timestamps for second review."""
from pathlib import Path
import os
import sys
sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import json
import tempfile
import time
from run_benchmark import HERE, ROOT, sha, dump
from app.config import Config
from app.pipeline.extract import extract_audio
from app.pipeline.transcribe import _load_model
import soundfile as sf

OUT = HERE / 'results/second'
OUT.mkdir(parents=True, exist_ok=True)
ffmpeg = next((ROOT / 'voice-analysis/.runtime/ffmpeg').rglob('ffmpeg.exe'))
cfg = Config(FFMPEG=str(ffmpeg), FFPROBE=str(ffmpeg.with_name('ffprobe.exe')), ASR_LOCAL_FILES_ONLY=True)
hashes = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'voice-analysis/app').rglob('*.py')}
dump(OUT / 'app-hashes-before.json', hashes)
for file_id in ['real_01', 'real_02', 'real_03', 'real_04']:
    case = json.loads((HERE / f'results/cases/{file_id}.json').read_text())['case']
    assert sha(case['path']) == case['source_sha256'], 'Source changed since first benchmark'
    wav = OUT / (file_id + '.wav')
    if not wav.exists():
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            decoded = extract_audio(case['path'], tmp, cfg)
            sf.write(wav, decoded['audio'], 16000, subtype='FLOAT')
    print('Decoded', file_id, flush=True)
model, provenance = _load_model(cfg.ASR_REPO, cfg.ASR_REVISION, cfg.ASR_LOCAL_PATH,
    cfg.ASR_CACHE_DIR, True, cfg.ASR_CPU_THREADS, cfg.ASR_COMPUTE_TYPE)
for file_id in ['real_02', 'real_04', 'real_01']:
    target = OUT / (file_id + '_asr.json')
    if target.exists():
        continue
    print('Whisper offline', file_id, flush=True)
    audio, _ = sf.read(OUT / (file_id + '.wav'), dtype='float32')
    started = time.perf_counter()
    segments, info = model.transcribe(audio, language='id', word_timestamps=True, beam_size=5,
        vad_filter=False, condition_on_previous_text=False, temperature=0)
    result = []
    for segment in segments:
        result.append({'start': segment.start, 'end': segment.end, 'text': segment.text,
                       'no_speech_prob': segment.no_speech_prob, 'avg_logprob': segment.avg_logprob,
                       'words': [{'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability}
                                 for w in segment.words or []]})
    dump(target, {'file_id': file_id, 'segments': result, 'model': provenance,
                  'runtime_seconds': time.perf_counter() - started,
                  'note': 'Full audio; no VAD cropping. ASR boundaries are approximate, not phonetic ground truth.'})
    print('Saved ASR', file_id, flush=True)
