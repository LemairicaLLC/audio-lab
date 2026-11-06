"""Test-signal generators. Every function returns a 1-D float64 NumPy array in the range -1..+1."""
import numpy as np

def db_to_amp(db):
    """Convert a level in dB (relative to full scale) to a linear amplitude."""
    return 10.0 ** (db / 20)

def sine(freq, duration, fs, level_dbfs=-6.0):
    """A pure sine wave.  level_dbfs is the peak level: 0 dBFS = full-scale peak. """
    t= np.arange(int(duration * fs)) / fs
    return db_to_amp(level_dbfs) * np.sin(2 * np.pi * freq * t)

def log_sweep(f1, f2, duration, fs, level_dbfs=-6.0):
    """Exponential (log) sine sweep from f1 to f2 Hz, plus its inverse filter.
    Convolving a recording of the sweep with the inverse filter gives the 
    system's impulse response (the Farina method). 
    """
    n = int(duration * fs)
    t = np.arange(n) / fs
    k =  np.log(f2 / f1)
    sweep = np.sin(2 * np.pi * f1 * duration / k * (np.exp(t * k / duration) - 1))
    sweep *= db_to_amp(level_dbfs)
    # Inverse filter: time-reversed sweep with a -6 dB/octave amplitude correction
    envelope = np.exp(-t * k / duration)
    inverse = sweep[::-1] * envelope
    inverse /= np.sum(sweep*inverse[::-1]) #normalize so sweep*inverse = unit impulse
    return sweep, inverse

def white_noise(duration, fs, level_dbfs=-20.0, seed=0):
    """Gaussian white noise at the given RMS level (dBFS)."""
    rng = np.random.default_rng(seed)
    return db_to_amp(level_dbfs) * rng.standard_normal(int(duration * fs))

def add_noise(x, snr_db, seed=0):
    """Add white noise to a x so that signal to noise ratio equals snr_db."""
    rng = np.random.default_rng(seed)
    p_signal = np.mean(x ** 2)
    p_noise = p_signal / (10 ** (snr_db / 10))
    return x + np.sqrt(p_noise) * rng.standard_normal(x.shape)
