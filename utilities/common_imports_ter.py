import sys,os,pickle
num_threads = 7
os.environ["MKL_NUM_THREADS"] = "%s"%num_threads
os.environ["NUMEXPR_NUM_THREADS"] = "%s"%num_threads
os.environ["OMP_NUM_THREADS"] = "%s"%num_threads
os.environ["OPENBLAS_NUM_THREADS"] = "%s"%num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "%s"%num_threads
os.environ["NUMBA_NUM_THREADS"] = "%s"%num_threads

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import importlib
