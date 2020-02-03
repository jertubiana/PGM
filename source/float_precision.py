import numpy as np

double_precision = False

if double_precision:
    curr_float = np.float64
    curr_int = np.int64
else:
    curr_float = np.float32
    curr_int = np.int16
