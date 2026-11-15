""" Step 5: simulate a 4-mic array, steer a delay-and-sum beamformer, measure the benefit"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audiolab import beamform, signals
from audiolab.array_sim import simulate_ula

FS = 16000   #voice-band sample rate, typical for voice front ends
N_MICS = 4
SPACING = 0.04  # 4cm between mics
TARGET_DEG = 0  #talker straight ahead
NOISE_DEG = 50  #interfering noise source off to the side
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

# A voice-like target: harmonics of 150Hz with a 4Hz syllable-rate envelope.
t = np.arange(2 * FS) / FS
target = sum(np.sin(2 * np.pi * 150 * h * t) / h for h in range(1,20))
target *= 0.5 * (1+ np.sin(2 * np.pi * 4 * t))
target *= 0.1 / np.max(np.abs(target))
interferer = signals.white_noise(2.0, FS, level_dbfs=-26, seed=3)

# The beamformer is linear, so run target and interferer through it separately to score them.
mic_target = simulate_ula([(target, TARGET_DEG)], FS, N_MICS, SPACING, self_noise_dbfs=-200)
mic_noise = simulate_ula([(interferer, NOISE_DEG)], FS, N_MICS, SPACING, self_noise_dbfs=-60)

def sir_db(sig, noise):
    return 10 * np.log10(np.mean(sig ** 2) / np.mean(noise **2))

sir_in = sir_db(mic_target[0], mic_noise[0])
out_t = beamform.delay_and_sum(mic_target, FS, SPACING, TARGET_DEG)
out_n = beamform.delay_and_sum(mic_noise, FS, SPACING, TARGET_DEG)
sir_out = sir_db(out_t, out_n)

fig, ax = plt.subplots(figsize=(8,4))
for f in (500, 1000, 2000, 4000):
    a, r = beamform.beam_pattern(N_MICS, SPACING, f, steer_deg=TARGET_DEG)
    ax.plot(a, r, label=f"{f} Hz")
ax.axvline(NOISE_DEG, color="k", ls="--", lw=1, label="noise source")
ax.set(xlabel="Arrival angle (deg)", ylabel="Response (dB)", ylim=(-30, 2),
       title=f"Delay-and-sum beam pattern: {N_MICS} mics, {SPACING*100:.0f} cm spacing")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(REPORTS / "beam_pattern.png", dpi=120)

results = {"sir_in_db": round(sir_in, 1), "sir_out_db": round(sir_out, 1),
           "improvement_db": round(sir_out - sir_in, 1)}
(REPORTS / "beamform.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))

