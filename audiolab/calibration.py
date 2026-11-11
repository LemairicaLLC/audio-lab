""" Load and apply a measurement microphone calibration file.
Typical calibration files are plain text: optional header lines, then one "frequency_hz correction_db" pair per line.  The correction is how much the mic over-read at that frequency,
so we must subtract it.  """

import numpy as np

def load_cal(path):
    freqs, dbs = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.replace(",", "").split()
            if len(parts) < 2:
                continue
            try:
                freqs.append(float(parts[0]))
                dbs.append(float(parts[1]))
            except ValueError:
                continue #header or comment line        
    return np.array(freqs), np.array(dbs)

def apply_cal(freqs_hz, response_db, cal_freqs, cal_db):
    """ return response_db corrected by the calibration curve. """
    correction = np.interp(freqs_hz, cal_freqs, cal_db, left=cal_db[0], right=cal_db[-1])
    return response_db - correction
