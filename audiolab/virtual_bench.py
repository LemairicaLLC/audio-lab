""" A software stand-in for a speaker, a room and a measurment mic.

play_and_record() has the same shape as sounddevice.playrec(), so a script can switch between
real hardware and this virtual bench with one flag.
Every imprefection below is deliberate, so your measurements have something to find.
"""

import numpy as np

FS_DEFAULT = 48000
LATENCY_S = 0.003 #3ms: about 1m of air between speaker and mic
NOISE_FLOOR_DBFS = -75.0 # room plus mic self-noise
MIC_BUMP_DB = 3.0 # the virtual mic over-reads by 3dB around 8kHz
MIC_BUMP_HZ = 8000.0

def _speaker_db(f):
    """ speaker magnitude response: 2nd order high pass at 80Hz, low pass at 12 kHz"""
    f = np.maximum(f, 1.0)
    hp = (f / 80) ** 2 / np.sqrt(1 + (f / 80)** 4)
    lp = 1 / np.sqrt(1+ (f / 12000) **4)
    return 20 * np.log10(hp * lp)

def mic_error_db(f):
    """ how far the virtual mic deviates from flat (this is what the cal file corrects) """
    f = np.maximum(f, 1.0)
    return MIC_BUMP_DB * np.exp(-np.log2(f / MIC_BUMP_HZ) **2 / (2*0.35 ** 2))


def _room_ir(fs, seed):
    """ Direct sound plus a decaying reverberant tail (about 15dB below the direct sound) """
    rng = np.random.default_rng(seed)
    n= int(0.25 *fs)
    t = np.arange(n)/fs
    tail = rng.standard_normal(n) * np.exp(-t / 0.04) * 10 ** (-45/20)
    tail[: int(0.002*fs)] = 0 # first reflection arrives 2ms after the direct sound
    ir = tail
    ir[0] = 1.0
    return ir

def play_and_record(x, fs=FS_DEFAULT, seed=0):
    """play x through a virtual speaker into the virtual room and mic; return the recording. """
    x = np.asarray(x, dtype=float).ravel()
    y = x + 0.01*x **2 - 0.03*x **3      # 1. mild 2nd and 3rd harmonic distortion
    room = _room_ir(fs, seed)            # 2. room reflections
    n = len(y) + len(room)
    nfft = 1 << (n-1).bit_length()
    f = np.fft.rfftfreq(nfft, 1 / fs)
    H = 10 **((_speaker_db(f) + mic_error_db(f)) / 20) # 3. speaker + mic frequency response
    H = H * np.exp(-2j * np.pi * f * LATENCY_S)        # 4. Time of flight
    Y = np.fft.rfft(y, nfft) * np.fft.rfft(room, nfft) * H
    rec = np.fft.irfft(Y, nfft)[: len(x)]
    rng = np.random.default_rng(seed+1)
    rec += 10 ** (NOISE_FLOOR_DBFS / 20) * rng.standard_normal(len(rec))  # 5. noise floor 
    return rec

def write_virtual_cal(path):
    """Write a calibration file for the virtual mic, in the usual 'Freq dB' text format """
    freqs = np.geomspace(10, 24000, 200)
    with open(path, "w", encoding = "utf-8") as fh:
        fh.write(' "Virtual reference mic, Sens Factor =0.0dB, SERNO: VIRTUAL-001" \n')
        for fr, db in zip(freqs, mic_error_db(freqs)):
            fh.write(f"{fr:.2f}\t{db:.3f}\n")



