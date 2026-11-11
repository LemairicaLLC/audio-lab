""" Step 3: play a sweep and a tone, recorde them, and measure the result.

real hardware (PC with the USB reference mic):
    uv run python -m scripts.measure_speaker --in 3 --out 5 --cal data/my_mic_cal.txt
virtual bench (any computer, no hardware):
    uv run python -m scripts.measure_speaker --virtual
    """

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")    #save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from audiolab import calibration, metrics, signals

FS = 48000
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

# Before (incorrect)
# p = argparse.ArgumentParser
# p.add_argument("--virtual", action="store_true", help="use the software bench instead of hardware")
# p.add_argument("--in", dest="dev_in", type=int, help="input device number (from query devices)") 
# p.add_argument("--out", dest="dev_out", type=int, help="output device number")
# p.add_argument("--cal", help="mic calibration file")
# args = p.parse_args()

p = argparse.ArgumentParser(description="Measure speaker frequency response and distortion.")
p.add_argument("--virtual", action="store_true", help="use the software bench instead of hardware")
p.add_argument("--in", dest="dev_in", type=int, help="input device number (from query devices)") 
p.add_argument("--out", dest="dev_out", type=int, help="output device number")
p.add_argument("--cal", help="mic calibration file")
args = p.parse_args()



if args.virtual:
    from audiolab import virtual_bench

    def play_and_record(x):
        return virtual_bench.play_and_record(x, FS)

if not args.cal:
    args.cal = "data/virtual_mic_cal.txt"
    Path("data").mkdir(exist_ok=True)
    virtual_bench.write_virtual_cal(args.cal)

else:
    import sounddevice as sd

    def play_and_record(x):
        rec = sd.playrec(x, samplerate=FS, channels = 1, 
                        device=(args.dev_in, args.dev_out), blocking = True)
        return rec[:, 0]

pad = np.zeros(int(0.5* FS))

# ---Frequency response from a log sweep ---
sweep, inverse = signals.log_sweep(20, 20000, 3.0, FS, level_dbfs=-12)
rec_sweep = play_and_record(np.concatenate([pad, sweep, pad]))
sf.write(REPORTS / "rec_sweep.wav", rec_sweep, FS, subtype="PCM_24")
ir = metrics.impulse_response(rec_sweep, inverse)
latency_ms = (np.argmax(np.abs(ir)) - (len(inverse)-1) - len(pad)) / FS*1000
freqs, mag = metrics.frequency_response(ir, FS)
mag -= np.interp(1000, freqs, mag) # normalise to 0dB at 1kHz

if args.cal:
    cf, cdb = calibration.load_cal(args.cal)
    mag_cal = calibration.apply_cal(freqs, mag, cf, cdb)
else:
    mag_cal = mag

# --- Distortion from a steady tone
tone = signals.sine(997, 2.0, FS, level_dbfs=-6)
rec_tone = play_and_record(np.concatenate([pad, tone, pad]))
steady = rec_tone[len(pad) + FS // 4: len(pad) + len(tone) - FS // 4]  #skip start/stop transients
thd_db, f0 = metrics.thd(steady, FS)
thdn_db = metrics.thd_n(steady, FS)

# --- Plots ---
fig, ax = plt.subplots(figsize=(8,4))
band = (freqs >= 20) & (freqs<= 20000)
ax.semilogx(freqs[band], mag[band], label = "raw", alpha=0.6)
ax.semilogx(freqs[band], mag_cal[band], label="mic-calibrated")
ax.set(xlabel= "Frequency (Hz)", ylabel="dB re 1kHz", ylim=(-40, 10),
       title="Speaker frequency response" + (" (virtual bench)" if args.virtual else ""))
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(REPORTS / "frequency_response.png", dpi=120)

results = {"virtual": args.virtual, "latency_ms": round(latency_ms, 2), 
           "tone_hz": round(f0, 1), "thd_db": round(thd_db, 1), "thdn_db": round(thdn_db, 1), 
           "level_dbfs": round(metrics.rms_dbfs(steady), 1)}
(REPORTS / "speaker.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent = 2))
print("Saved", REPORTS / "frequency_response.png")
