"""Step 6: gather every step's results into one Markdown summary"""

import json
from datetime import datetime
from pathlib import Path

R = Path("reports")

def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None

spk, rtp, bf = load("speaker"), load("rtp"), load("beamform")
lines = [f"# Audio Lab Report", f"Generated {datetime.now():%Y-%m-%d %H:%M}", ""]

if spk:
    src = "virtual bench" if spk["virtual"] else "hardware"
    lines += ["## Speaker measurement (" + src + ")",
              f"- Latency: {spk['latency_ms']} ms",
              f"- THD: {spk['thd_db']} dB, THD+N: {spk['thdn_db']} dB at {spk['tone_hz']} Hz",
              "- ![Frequency response](frequency_response.png)", ""]

if rtp:
    lossy_data = rtp.get('lossy', {})
    lost_packets = lossy_data.get('lost', lossy_data.get('lost_seq', 'N/A'))
    thdn_val = lossy_data.get('thdn_db', 'N/A')
    
    lines += ["## RTP / AES67 capture",
              f"- Clean: {rtp.get('clean', {}).get('packets_received', 'N/A')} packets, THD+N {rtp.get('clean', {}).get('thdn_db', 'N/A')} dB",
              f"- Lossy: lost packets {lost_packets}, THD+N {thdn_val} dB",  ""]

if bf:
    lines += ["## Beamformer",
              f"- Signal-to-interference in: {bf['sir_in_db']} dB, out: {bf['sir_out_db']} dB "
              f"(improvement {bf['improvement_db']} dB)",
              "- ![Beam pattern](beam_pattern.png)", ""]

# Un-indented to root level so it always writes regardless of which JSON files exist
(R / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))