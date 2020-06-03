"""
 Copyright 2020 - by Jerome Tubiana (jertubiana@gmail.com)
     All rights reserved

     Permission is granted for anyone to copy, use, or modify this
     software for any uncommercial purposes, provided this copyright
     notice is retained, and note is made of any changes that have
     been made. This software is distributed without any warranty,
     express or implied. In no event shall the author or contributors be
     liable for any damage arising out of the use of this software.

     The publication of research using this software, modified or not, must include
     appropriate citations to:
"""

import sys,os,pickle
def set_num_threads(num_threads=2):
    os.environ["MKL_NUM_THREADS"] = "%s"%num_threads
    os.environ["NUMEXPR_NUM_THREADS"] = "%s"%num_threads
    os.environ["OMP_NUM_THREADS"] = "%s"%num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = "%s"%num_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = "%s"%num_threads
    os.environ["NUMBA_NUM_THREADS"] = "%s"%num_threads

if __name__ == '__main__':
    set_num_threads()
