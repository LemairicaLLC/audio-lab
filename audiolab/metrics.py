""" Audio performance metrics: level, THD, THD+N, frequency response, delay. """
import numpy as np

def blackman_harris(n):
    """4-term Blackman-Harris window: very low sidelobes (about -92 dB), good for THD"""
    a = [0.35875, 0.48829, 0.14128, 0.01168]
    k = np.arange(n)
    return (a[0] - a[1] * np.cos(2 * np.pi * k / (n - 1))
            + a[2] * np.cos(4 * np.pi * k / (n - 1))
            - a[3] * np.cos(6 * np.pi * k / (n - 1)))

def rms_dbfs(x):
    """RMS level in dBFS, using the AES17 convention: a full-scale sine wave reads 0dBFS """
    rms = np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))
    return 20 * np.log10(rms*np.sqrt(2) + 1e-20)  # add a tiny number to avoid log(0) for silence

def spectrum(x, fs):
    """Windowed power spectrum.  Returns (freqs_hz, power) for the positive frequencies """
    win = blackman_harris(len(x))
    X = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    return freqs, np.abs(X) ** 2 

def _band(freqs, f_lo, f_hi):
    """Return the indices of the frequencies in the range f_lo..f_hi"""
    return (freqs >= f_lo) & (freqs <= f_hi)

def thd_n(x, fs, f_lo=20.0, f_hi=20000.0, notch_bins=8):
    """ THD+N in dB, everything except the fundamental, relative to the fundamental. 
     Works like an analyzer's notch filter: it finds the biggest peak, removes the bins around it, and sums the rest of the power in the band.  """
    freqs, p = spectrum(x, fs)
    band = _band(freqs, f_lo, min(f_hi, fs/2))
    k0 = np.argmax(np.where(band, p, 0))  # index of the fundamental
    fund = slice(max(0, k0-notch_bins, 0), k0+notch_bins+1)  # slice of the fundamental and its neighbors
    p_fund = p[fund].sum()
    residual = p.copy()
    residual[fund] = 0
    p_rest = residual[band].sum()
    return 10 * np.log10(p_rest / p_fund)

def thd(x, fs, n_harmonics=5, notch_bins=8):
    """ THD in dB, power in harmonics 2...n+1, relative to the fundamental. (Noise excluded). """
    freqs, p = spectrum(x, fs)
    df = freqs[1]
    k0 = np.argmax(p[1:]) + 1  # index of the fundamental, skip DC
    f0 = freqs[k0]

    def power_at(f):
        k = int(round(f / df))
        return p[max(k-notch_bins, 0):k+notch_bins+1].sum()

    p_fund = power_at(f0)
    harmonics = [power_at(h * f0) for h in range(2, n_harmonics+2) if h * f0 < fs/2]
    return 10 * np.log10(sum(harmonics) / p_fund), f0

def impulse_response(recorded, inverse_filter):
    """Deconvolve the recorded log-sweep with its inverse filter (FFT convolution)."""
    n = len(recorded) + len(inverse_filter) - 1
    nfft = 1 << (n - 1).bit_length()  # next power of 2
    h = np.fft.irfft(np.fft.rfft(recorded, nfft) * np.fft.rfft(inverse_filter, nfft), nfft)
    return h[:n]  # truncate to the correct length

def frequency_response(ir, fs, n_fft=None, peak_window=4096): 
    """Magnitude response in dB of an impulse response using a window around its peak """
    k = np.argmax(np.abs(ir))  # index of the peak
    start = max(k - 64, 0)
    seg = ir[start:start + peak_window]
    n_fft = n_fft or 1 << (len(seg) - 1).bit_length()  # next power of 2
    H = np.fft.rfft(seg, n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    return freqs, 20 * np.log10(np.abs(H) + 1e-12)  # add a tiny number to avoid log(0) for silence

def gcc_phat_delay(sig,ref, fs, max_delay_s=None):
    """Time delay of a sig relative to a reference signal ref using the GCC-PHAT method. Returns delay in seconds.
    Positive result means that sig arrives later than ref """
    n = len(sig) + len(ref) 
    nfft = 1 << (n - 1).bit_length()  # next power of 2
    S = np.fft.rfft(sig, nfft)
    R = np.fft.rfft(ref, nfft)
    cross = S * np.conj(R)
    cc = np.fft.irfft(cross / (np.abs(cross) + 1e-12), nfft)  # add a tiny number to avoid division by zero
    max_shift = nfft // 2 if max_delay_s is None else int(max_delay_s * fs)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))  # wrap around
    k = np.argmax(np.abs(cc))
    # Parabolic interpolation for sub-sample accuracy
    if 0 < k < len(cc) - 1:
        a, b, c = np.abs(cc[k-1]), np.abs(cc[k]), np.abs(cc[k+1])
        k = k + 0.5 * (a - c) / (a - 2*b + c)
    return (k - max_shift) / fs






