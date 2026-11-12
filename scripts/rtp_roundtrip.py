""" Step 4: packetise audio into an AES67-style RTP capture, then decode it like a log file."""

import json
from pathlib import Path

import numpy as np

from audiolab import metrics, rtp_pcap, signals
FS = 48000
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
DATA = Path("data")
DATA.mkdir(exist_ok=True)

tone = signals.sine(997, 1.0, FS, level_dbfs=-6)

clean = DATA / "aes67_clean.pcap"
lossy = DATA / "aes67_lossy.pcap"
n = rtp_pcap.write_rtp_pcap(clean, tone, FS)
rng = np.random.default_rng(7)
dropped = sorted(rng.choice(n, size=5, replace=False).tolist())  # lose 5 of 1000 packets
rtp_pcap.write_rtp_pcap(lossy, tone, FS, drop_packets=dropped)

results = {}
for name, path in [("clean", clean), ("lossy", lossy)]:
    audio, stats = rtp_pcap.read_rtp_pcap(path)
    results[name] = {"packets_received": stats["received"], "packets_expected": stats["expected"],
                     "lost seq": stats["lost"], "thdn_db": round(metrics.thd_n(audio, FS), 1)}
    print(f"{name:6s} received {stats['received']}/{stats['expected']} packets, "
          f"lost {stats['lost']}, THD+N {results[name]['thdn_db']} dB")
(REPORTS / "rtp.json").write_text(json.dumps(results, indent=2))
print ("Open", clean, "in Wireshrak to inspect the packets.")