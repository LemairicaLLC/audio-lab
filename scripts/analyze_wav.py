"""Step 2: measure level, THD and THD+N of any WAV file.
Usage: uv run python -m scripts.analyze_wav data/tone_997hz.wav
"""
import sys

import soundfile as sf

from audiolab import metrics

path = sys.argv[1]
x, fs = sf.read(path, dtype="float64")
if x.ndim > 1:
    x = x[:,0] #first channel only
thd_db, f0 = metrics.thd(x, fs)
print(f"File:      {path} ") 
print(f"Rate:      {fs} Hz, {len(x) / fs:.2f} s")
print(f"Level:     {metrics.rms_dbfs(x):6.1f} dBFS")
print(f"Tone:      {f0:6.1f} Hz")
print(f"THD:       {thd_db:6.1f} dB")
print(f"THD+N:     {metrics.thd_n(x, fs):6.1f} dB")
