import numpy as np

from audiolab import beamform
from audiolab.array_sim import simulate_ula

FS = 16000

def test_delay_and_sum_reduces_uncorrelated_noise():
    silence = np.zeros(FS)
    noise_only = simulate_ula([(silence, 0)], FS, n_mics=4, spacing_m=0.04, self_noise_dbfs=-30)
    out = beamform.delay_and_sum(noise_only, FS, 0.04, steer_deg=30)
    gain_db = 10 * np.log10(np.mean(noise_only[0] ** 2) / np.mean(out ** 2))
    # Averaging 4 mics with independent noise -> 10*log10(4) = 6 dB less noise
    assert 5 < gain_db < 7

def test_beam_pattern_peak_at_steer_angle():
    angles, resp = beamform.beam_pattern(4, 0.04, 2000, steer_deg=20)
    assert abs(angles[np.argmax(resp)] - 20) <= 1




