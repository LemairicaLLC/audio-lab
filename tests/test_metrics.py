import numpy as np

from audiolab import metrics, signals

FS = 48000

def test_full_scale_sine_reads_0_dbfs():
    x = signals.sine(1000, 1.0, FS, level_dbfs=0.0)
    assert abs(metrics.rms_dbfs(x)) < 0.01

def test_thd_of_known_harmonic():
    # 1% second harmonic = -40 dB THD
    fund = signals.sine(997, 1.0, FS, level_dbfs=-6.0)
    x = fund + 0.01 * signals.sine(2*997, 1.0, FS, level_dbfs=-6.0)
    value, f0 = metrics.thd(x, FS)
    assert abs(value - (-40.0)) < 0.2
    assert abs(f0 - 997) < 2

def test_thdn_tracks_added_noise():
    x = signals.add_noise(signals.sine(997, 1.0, FS), snr_db=60)
    # Only part of the white noise falls inside the 20Hz-20kHz band of a 24kHz span, so allow ~1 dB tolerance         
    assert -62 < metrics.thd_n(x, FS) < -59  # THD+N is worse than THD because of noise floor

def test_gcc_phat_finds_delay():
    from audiolab.array_sim import fractional_delay
    ref = signals.white_noise(0.5, FS)
    sig = fractional_delay(ref, 0.00025, FS)  #  250us delay = 12 samples at 48kHz
    assert abs(metrics.gcc_phat(sig, ref, FS, max_delay_s=0.01) - 0.00025) < 1e-5

def test_sweep_deconvolution_gives_flat_response():
    sweep, inverse = signals.log_sweep(20, 20000, 2.0, FS)
    ir = metrics.impulse_response(sweep, inverse)
    f, mag = metrics.frequency_response(ir, FS)
    band = (f > 100) & (f < 10000)
    assert np.ptp(mag[band]) < 1  # peak-to-peak ripple (flat) 1 dB in the passband

