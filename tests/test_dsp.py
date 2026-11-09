import numpy as np
import pytest

# Import your actual THD function once it's implemented in your project structure
# from audio_lab.analysis import calculate_thd

def test_thd_pure_tone_with_third_harmonic():
    """
    Test that a 997 Hz tone with a 0.1% third harmonic yields a THD of -60 dB.
    """
    sr = 48000          # Standard audio sample rate (Hz)
    duration = 2.0      # Duration in seconds for good frequency resolution
    f_fund = 997.0      # Fundamental frequency (Hz)
    f_harm = 3 * f_fund # Third harmonic frequency (2991.0 Hz)
    
    # Time vector
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Generate signal: Fundamental + 0.1% (-60 dB) third harmonic
    # Using a Hann window or a whole number of cycles avoids spectral leakage in FFT-based THD
    signal = np.sin(2 * np.pi * f_fund * t) + 0.001 * np.sin(2 * np.pi * f_harm * t)
    
    # Calculate THD (replace with your actual function call)
    # thd_db = calculate_thd(signal, sr, fundamental_freq=f_fund)
    
    # Placeholder assertion demonstrating expected tolerance 
    # (FFT-based measurements usually allow a tiny margin for windowing/noise floor)
    # assert thd_db == pytest.approx(-60.0, abs=0.2)