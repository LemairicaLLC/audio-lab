import numpy as np

from audiolab import rtp_pcap, signals

def test_124_roundtrip():
    x = np.array([0.0, 0.5, -0.5, 0.999, -1.0])
    y = rtp_pcap.l24_to_float(rtp_pcap.float_to_l24(x))
    assert np.max(np.abs(x-y)) < 1e-6

def test_pcap_roundtrip_with_loss(tmp_path):
    x = signals.sine(1000, 0.1, 48000)   #4800 samples = 100 packets
    path = tmp_path / "test.pcap"
    rtp_pcap.write_rtp_pcap(path, x, drop_packets=[10,11,50])
    audio, stats = rtp_pcap.read_rtp_pcap(path)
    assert stats["lost"] == [10,11,50]
    assert stats["received"] == 97
    good = np.ones(len(x), bool)
    for p in stats["lost"]:
        good[p*48:(p+1) * 48] = False
    assert np.max(np.abs(audio[good] - x[good])) < 1e-6
    assert np.all(audio[~good] == 0)