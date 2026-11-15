""" Delay and sum beamforming for a uniform linear array"""
import numpy as np

from .array_sim import SPEED_OF_SOUND, fractional_delay, mic_delays

def delay_and_sum(mics, fs, spacing_m, steer_deg):
    """Steer the array toward steer_deg: undo each mic's delay, then average"""
    delays = mic_delays(mics.shape[0], spacing_m, steer_deg)
    aligned = [fractional_delay(ch, -d, fs) for ch, d in zip(mics, delays)]
    return np.mean(aligned, axis=0)

def beam_pattern(n_mics, spacing_m, freq_hz, steer_deg=0.0, angles_deg=None, c=SPEED_OF_SOUND):
    """Theoretical array response (dB) versus arrival angle at one frequency"""
    if angles_deg is None:
        angles_deg = np.linspace(-90, 90, 361)
    pos = np.arange(n_mics) * spacing_m
    pos -= pos.mean()
    k = 2 * np.pi * freq_hz / c
    steer = np.exp(1j * k * pos * np.sin(np.radians(steer_deg)))
    resp = []
    for a in angles_deg:
        arrival = np.exp(-1j * k * pos * np.sin(np.radians(a)))
        resp.append(np.abs(np.sum(steer * arrival)) / n_mics)
    return angles_deg, 20 * np.log10(np.array(resp) + 1e-12)
