"""Simulate a uniform linear microphone array (ULA) hearing far-field sources."""
import numpy as np

SPEED_OF_SOUND = 343 #m/s at 20C

def fractional_delay(x, delay_s, fs):
    """Delay x by any amount of time (not just whole samples) using an FFT phase shift."""
    n = len(x)
    nfft = 1 << (n + int(abs(delay_s) * fs) + 64-1).bit_length()
    f = np.fft.rfftfreq(nfft, 1/fs)
    X = np.fft.rfft(x,nfft) * np.exp(-2j * np.pi * f *delay_s)
    return np.fft.irfft(X, nfft)[:n]

def mic_delays(n_mics, spacing_m, angle_deg, c=SPEED_OF_SOUND):
    """Arrival delay at each mic for a plane wave from angle_deg (0 = broadside)"""
    positions = np.arange(n_mics) * spacing_m
    positions -= positions.mean() #center the array on 0
    return positions * np.sin(np.radians(angle_deg)) / c

def simulate_ula(sources, fs, n_mics=4, spacing_m=0.04, self_noise_dbfs=-60, seed=1):
    """ Render what each mic hears.
    Sources: list of (signal, angle_deg) tuples, all the same lenght.
    Returns an array of shape (n_mics, n_samples.)"""

    n = len(sources[0][0])
    out = np.zeros((n_mics, n))
    for sig, angle in sources:
        for m, d in enumerate(mic_delays(n_mics, spacing_m, angle)):
            out[m] += fractional_delay(sig, d, fs)
    rng = np.random.default_rng(seed)
    out += 10 ** (self_noise_dbfs / 20) * rng.standard_normal(out.shape)
    return out
