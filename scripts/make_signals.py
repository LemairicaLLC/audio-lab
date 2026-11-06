""" Step 1: generate test signals and save them as 24-bit WAV files in data/."""
from pathlib import Path

import soundfile as sf

from audiolab import signals

FS = 48000
DATA = Path("data")
DATA.mkdir(exist_ok=True)

tone = signals.sine(997, 3.0, FS, level_dbfs=-6.0)
sweep, inverse = signals.log_sweep(20, 20000, 3.0, FS, level_dbfs=-12.0)
noise = signals.white_noise(3.0, FS, level_dbfs=-20.0)

sf.write(DATA / "tone_997hz.wav", tone, FS, subtype="PCM_24")
sf.write(DATA / "sweep_20_20k.wav", sweep, FS, subtype="PCM_24")
sf.write(DATA / "noise_white.wav", noise, FS, subtype="PCM_24")
print("Wrote tone, sweep and noise WAV files to", DATA.resolve())