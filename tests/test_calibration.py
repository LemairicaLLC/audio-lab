import numpy as np

from audiolab import calibration

def test_load_and_apply_cal(tmp_path):
    p = tmp_path / "mic.txt"
    p.write_text(' "Sens Factor =1.867dB, SERNO: 7022488" \n20 -1.0\n1000 0.0\n20000 2.0\n ')
    f, db = calibration.load_cal(p)
    assert list(f) == [20, 1000, 20000]
    corrected = calibration.apply_cal(np.array([1000, 20000]), np.array([0.0, 0.0]), f, db)
    assert np.allclose(corrected, [0.0, -2.0])
    