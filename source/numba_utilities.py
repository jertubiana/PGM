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

import numpy as np
from numba import njit, prange
from numba.extending import get_cython_function_address
import ctypes

from float_precision import double_precision, curr_float, curr_int

if double_precision:
    signature = "(float64[:])(int64[:,:],float64[:,:])"
else:
    signature = "(float32[:])(int16[:,:],float32[:,:])"


@njit(signature, parallel=True, cache=True, nogil=False)
def dot_Potts_C(config, fields):
    B = config.shape[0]
    N = config.shape[1]
    # q = fields.shape[-1]
    out = np.zeros(B, dtype=curr_float)
    for b in prange(B):
        for n in prange(N):
            out[b] += fields[n, config[b, n]]
    return out


if double_precision:
    signature = "(float64[:])(int64[:,:],float64[:,:,:])"
else:
    signature = "(float32[:])(int16[:,:],float32[:,:,:])"


@njit(signature, parallel=True, cache=True, nogil=False)
def dot_Potts2_C(config, fields):
    B = config.shape[0]
    N = config.shape[1]
    # q = fields.shape[-1]
    out = np.zeros(B, dtype=curr_float)
    for b in prange(B):
        for n in prange(N):
            out[b] += fields[b, n, config[b, n]]
    return out


if double_precision:
    signature = "(float64[:,:])(int64[:,:,:],float64[:,:,:])"
else:
    signature = "(float32[:,:])(int16[:,:,:],float32[:,:,:])"


@njit(signature, parallel=True, cache=True, nogil=False)
def dot_Potts3_C(config, fields):
    N_PT = config.shape[0]
    B = config.shape[1]
    N = config.shape[2]
    # q = fields.shape[-1]
    out = np.zeros((N_PT, B), dtype=curr_float)
    for n_pt in prange(N_PT):
        for b in prange(B):
            for n in prange(N):
                out[n_pt, b] += fields[n_pt, n, config[n_pt, b, n]]
    return out


if double_precision:
    signature = "(float64[:])(int64[:,:],int64[:,:],float64[:,:,:,:])"
else:
    signature = "(float32[:])(int16[:,:],int16[:,:],float32[:,:,:,:])"


def bilinear_form_Potts_C(X1, X2, couplings):
    B = X1.shape[0]
    N1 = couplings.shape[0]
    N2 = couplings.shape[1]
    out = np.zeros(B, dtype=curr_float)
    # Use a buffer to accumulate values in parallel.
    out_buffer = np.zeros([B, N1], dtype=curr_float)
    for b in prange(B):
        for n1 in prange(N1):
            for n2 in range(N2):
                out_buffer[b, n1] += couplings[n1, n2, X1[b, n1], X2[b, n2]]
    out = np.sum(out_buffer, 1)
    return out


if double_precision:
    signature = "(float64[:,:])(int64[:,:],float64[:,:,:],float64[:,:])"
else:
    signature = "(float32[:,:])(int16[:,:],float32[:,:,:],float32[:,:])"


@njit(signature, parallel=True, cache=True, nogil=False)
def compute_output_C(config, W, out):
    B = config.shape[0]
    N = config.shape[1]
    M = W.shape[0]
    # q = W.shape[2]

    # out = np.zeros( (M,B ) ,dtype=curr_float)
    out = out.T
    for m in prange(M):
        for b in prange(B):
            for n in prange(N):
                out[m, b] += W[m, n, config[b, n]]
    return out.T


if double_precision:
    signature = "(float64[:,:])(int64[:,:],float64[:,:,:],float64[:,:])"
else:
    signature = "(float32[:,:])(int16[:,:],float32[:,:,:],float32[:,:])"


@njit(signature, parallel=True, cache=True, nogil=False)
def compute_output_C2(config, W, out):
    B = config.shape[0]
    M = config.shape[1]
    N = W.shape[1]
    # q = W.shape[2]

    # out = np.zeros( (N,B) ,dtype=curr_float)
    out = out.T
    for m in prange(M):
        for n in prange(N):
            for b in prange(B):
                out[n, b] += W[m, n, config[b, m]]
    return out.T


if double_precision:
    signature = "(float64[:,:,:])(int64[:,:],float64[:,:,:,:],float64[:,:,:])"
else:
    signature = "(float32[:,:,:])(int16[:,:],float32[:,:,:,:],float32[:,:,:])"


@njit(signature, parallel=True, cache=True, nogil=False)
def compute_output_Potts_C(config, W, out):
    B = config.shape[0]
    M = W.shape[0]
    N = W.shape[1]
    q_up = W.shape[2]
    # q_down = W.shape[3]
    # out = np.zeros((q_up,M,B) ,dtype=curr_float)
    out = out.T

    for m in prange(M):
        for n in prange(N):
            for q_up_ in prange(q_up):
                for b in prange(B):
                    out[q_up_, m, b] += W[m, n, q_up_, config[b, n]]

    return out.T


if double_precision:
    signature = "(float64[:,:,:])(int64[:,:],float64[:,:,:,:],float64[:,:,:])"
else:
    signature = "(float32[:,:,:])(int16[:,:],float32[:,:,:,:],float32[:,:,:])"


@njit(signature, parallel=True, cache=True, nogil=False)
def compute_output_Potts_C2(config, W, out):
    B = config.shape[0]
    M = W.shape[0]
    N = W.shape[1]
    # q_up = W.shape[2]
    q_down = W.shape[3]
    # out = np.zeros((q_down,N,B) ,dtype=curr_float)
    out = out.T

    for m in prange(M):
        for n in prange(N):
            for q_down_ in prange(q_down):
                for b in prange(B):
                    out[q_down_, n, b] += W[m, n, config[b, m], q_down_]
    return out.T


if double_precision:
    signature = "(float64[:,:])(int64[:,:],int64)"
else:
    signature = "(float32[:,:])(int16[:,:],int64)"


@njit(signature, parallel=False, cache=True, nogil=False)
def average_C(config, q):
    B = config.shape[0]
    N = config.shape[1]
    out = np.zeros((N, q), dtype=curr_float)
    for b in prange(B):
        for n in prange(N):
            out[n, config[b, n]] += 1
    out /= B
    return out


if double_precision:
    signature = "(float64[:,:])(int64[:,:],float64[:],int64)"
else:
    signature = "(float32[:,:])(int16[:,:],float32[:],int64)"


@njit(signature, parallel=False, cache=True, nogil=False)
def weighted_average_C(config, weights, q):
    B = config.shape[0]
    N = config.shape[1]
    out = np.zeros((N, q), dtype=curr_float)
    for b in prange(B):
        for n in prange(N):
            out[n, config[b, n]] += weights[b]
    out /= weights.sum()
    return out


if double_precision:
    signature = "(float64[:,:,:])(float64[:,:],int64[:,:],int64)"
else:
    signature = "(float32[:,:,:])(float32[:,:],int16[:,:],int64)"


@njit(signature, parallel=True, cache=True, nogil=False)
def average_product_FxP_C(config1, config2, q):
    B = config1.shape[0]
    M = config1.shape[1]
    N = config2.shape[1]
    out = np.zeros((M, N, q), dtype=curr_float)
    for m in prange(M):
        for n in prange(N):
            for b in prange(B):
                out[m, n, config2[b, n]] += config1[b, m]
    out /= B
    return out


if double_precision:
    signature = "(float64[:,:,:,:])(int64[:,:],int64[:,:],int64,int64)"
else:
    signature = "(float32[:,:,:,:])(int16[:,:],int16[:,:],int64,int64)"


@njit(signature, parallel=True, cache=True, nogil=False)
def average_product_PxP_C(config1, config2, q1, q2):
    B = config1.shape[0]
    M = config1.shape[1]
    N = config2.shape[1]
    out = np.zeros((M, N, q1, q2), dtype=curr_float)
    for m in prange(M):
        for n in prange(N):
            for b in prange(B):
                out[m, n, config1[b, m], config2[b, n]] += 1
    out /= B
    return out


if double_precision:
    signature = "(float64[:,:,:,:])(int64[:,:],int64[:,:],float64[:],int64,int64)"
else:
    signature = "(float32[:,:,:,:])(int16[:,:],int16[:,:],float32[:],int64,int64)"


@njit(signature, parallel=True, cache=True, nogil=False)
def weighted_average_product_PxP_C(config1, config2, weights, q1, q2):
    B = config1.shape[0]
    M = config1.shape[1]
    N = config2.shape[1]
    out = np.zeros((M, N, q1, q2), dtype=curr_float)
    for m in prange(M):
        for n in prange(N):
            for b in prange(B):
                out[m, n, config1[b, m], config2[b, n]] += weights[b]
    out /= weights.sum()
    return out


if double_precision:
    signature = "(float64[:,:,:])(float64[:,:],int64[:,:],float64[:],int64)"
else:
    signature = "(float32[:,:,:])(float32[:,:],int16[:,:],float32[:],int64)"


@njit(signature, parallel=True, cache=True, nogil=False)
def weighted_average_product_FxP_C(config1, config2, weights, q):
    B = config1.shape[0]
    M = config1.shape[1]
    N = config2.shape[1]
    out = np.zeros((M, N, q), dtype=config1.dtype)
    for m in prange(M):
        for n in prange(N):
            for b in prange(B):
                out[m, n, config2[b, n]] += config1[b, m] * weights[b]
    out /= weights.sum()
    return out


if double_precision:
    signature = "(int64[:,:])(float64[:,:,:],float64[:,:],int64[:,:])"
else:
    signature = "(int16[:,:])(float32[:,:,:],float64[:,:],int16[:,:])"


@njit(signature, parallel=True)
def tower_sampling_C(cum_probabilities, rng, out):
    B = cum_probabilities.shape[0]
    N = cum_probabilities.shape[1]
    q = cum_probabilities.shape[2]
    for b in prange(B):
        for n in prange(N):
            low = 0
            high = q
            while low < high:
                middle = (low + high) // 2
                if rng[b, n] < cum_probabilities[b, n, middle]:
                    high = middle
                else:
                    low = middle + 1
            out[b, n] = high
    return out


if double_precision:
    signature = "(float64[:,:])(int64[:],float64[:,:],float64[:,:])"
else:
    signature = "(float32[:,:])(int16[:],float32[:,:],float32[:,:])"


@njit(signature, parallel=False, cache=True, nogil=False)
def substitute_0C(config, fields, out):
    M, q = fields.shape
    B = config.shape[0]
    for b in prange(B):
        for m in prange(M):
            out[b, m] = fields[m, config[b]]
    return out


if double_precision:
    signature = "(float64[:,:,:])(int64[:],float64[:,:,:],float64[:,:,:])"
else:
    signature = "(float32[:,:,:])(int16[:],float32[:,:,:],float32[:,:,:])"


@njit(signature, parallel=False, cache=True, nogil=False)
def substitute_1C(config, weights, out):
    M, q_in, q_out = weights.shape
    B = config.shape[0]
    for b in prange(B):
        for m in prange(M):
            for q_out_ in prange(q_out):
                out[b, m, q_out_] = weights[m, config[b], q_out_]
    return out


if double_precision:
    signature = "(float64[:,:])(float64[:,:,:],int64[:,:])"
else:
    signature = "(float32[:,:])(float32[:,:,:],int16[:,:])"


@njit(signature, parallel=False, cache=True, nogil=False)
def substitute_C(fields, config):
    B = fields.shape[0]
    N = fields.shape[1]
    # q = fields.shape[2]
    out = np.zeros((B, N), dtype=curr_float)
    for b in prange(B):
        for n in prange(N):
            out[b, n] = fields[b, n, config[b, n]]
    return out


addr_erfcx = get_cython_function_address(
    "scipy.special.cython_special", "__pyx_fuse_1erfcx")
addr_erf = get_cython_function_address(
    "scipy.special.cython_special", "__pyx_fuse_1erf")
addr_ndtri = get_cython_function_address(
    "scipy.special.cython_special", "ndtri")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
erfcx_fn = functype(addr_erfcx)
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
erf_fn = functype(addr_erf)
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
ndtri_fn = functype(addr_ndtri)

if double_precision:
    signature0 = "float64(float64)"
    signature1 = "float64[:](float64[:])"
    signature2 = "float64[:,:](float64[:,:])"
    signature3 = "float64[:,:,:](float64[:,:,:])"
else:
    signature0 = "float32(float32)"
    signature1 = "float32[:](float32[:])"
    signature2 = "float32[:,:](float32[:,:])"
    signature3 = "float32[:,:,:](float32[:,:,:])"


p = np.array(0.47047, dtype=curr_float)
pbis = np.array(0.332672, dtype=curr_float)
a1 = np.array(0.3480242, dtype=curr_float)
a2 = np.array(- 0.0958798, dtype=curr_float)
a3 = np.array(0.7478556, dtype=curr_float)
sqrtpiover2 = np.array(1.25331, dtype=curr_float)
log2 = np.array(0.6931471806, dtype=curr_float)
logsqrtpiover2 = np.array(0.2257913526, dtype=curr_float)
invsqrt2 = np.array(0.7071067812, dtype=curr_float)
sqrt2 = np.array(1.4142135624, dtype=curr_float)
sqrt2_double = np.array(1.4142135624, dtype=np.float64)
# logsqrtpiover2_double = np.array(0.2257913526,dtype=np.float64)


# See explanation in http://people.math.sfu.ca/~cbm/aands/page_299.htm
@njit(signature0, parallel=False)
def erf_numba0(x):
    t = 1 / (1 + p * np.abs(x))
    return np.sign(x) * (1 - t * (a1 + a2 * t + a3 * t**2) * np.exp(-x**2))


# See explanation in http://people.math.sfu.ca/~cbm/aands/page_299.htm
@njit(signature1, parallel=True)
def erf_numba1(x):
    t = 1 / (1 + p * np.abs(x))
    return np.sign(x) * (1 - t * (a1 + a2 * t + a3 * t**2) * np.exp(-x**2))


# See explanation in http://people.math.sfu.ca/~cbm/aands/page_299.htm
@njit(signature2, parallel=True)
def erf_numba2(x):
    t = 1 / (1 + p * np.abs(x))
    return np.sign(x) * (1 - t * (a1 + a2 * t + a3 * t**2) * np.exp(-x**2))


# See explanation in http://people.math.sfu.ca/~cbm/aands/page_299.htm
@njit(signature3, parallel=True)
def erf_numba3(x):
    t = 1 / (1 + p * np.abs(x))
    return np.sign(x) * (1 - t * (a1 + a2 * t + a3 * t**2) * np.exp(-x**2))


@njit(signature0, parallel=False)
def erf_times_gauss_numba0(x):
    if x < -6:
        out = 2 * np.exp(x**2 / 2)
    elif x > 0:
        t = 1 / (1 + pbis * x)
        out = t * (a1 + a2 * t + a3 * t**2)
    else:
        t = 1 / (1 - pbis * x)
        out = -t * (a1 + a2 * t + a3 * t**2) + 2 * np.exp(x**2 / 2)
    return sqrtpiover2 * out


@njit(signature1, parallel=True)
def erf_times_gauss_numba1(x):
    out = np.zeros(x.shape, dtype=curr_float)
    for i in prange(x.shape[0]):
        x_ = x[i]
        if x_ < -6:
            out[i] = 2 * np.exp(x_**2 / 2)
        elif x_ > 0:
            t = 1 / (1 + pbis * x_)
            out[i] = t * (a1 + a2 * t + a3 * t**2)
        else:
            t = 1 / (1 - pbis * x_)
            out[i] = -t * (a1 + a2 * t + a3 * t**2) + 2 * np.exp(x_**2 / 2)
    return sqrtpiover2 * out


@njit(signature2, parallel=True)
def erf_times_gauss_numba2(x):
    out = np.zeros(x.shape, dtype=curr_float)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            x_ = x[i, j]
            if x_ < -6:
                out[i, j] = 2 * np.exp(x_**2 / 2)
            elif x_ > 0:
                t = 1 / (1 + pbis * x_)
                out[i, j] = t * (a1 + a2 * t + a3 * t**2)
            else:
                t = 1 / (1 - pbis * x_)
                out[i, j] = -t * (a1 + a2 * t + a3 * t**2) + \
                    2 * np.exp(x_**2 / 2)
    return sqrtpiover2 * out


@njit(signature3, parallel=True)
def erf_times_gauss_numba3(x):
    out = np.zeros(x.shape, dtype=curr_float)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                x_ = x[i, j, k]
                if x_ < -6:
                    out[i, j, k] = 2 * np.exp(x_**2 / 2)
                elif x_ > 0:
                    t = 1 / (1 + pbis * x_)
                    out[i, j, k] = t * (a1 + a2 * t + a3 * t**2)
                else:
                    t = 1 / (1 - pbis * x_)
                    out[i, j, k] = -t * \
                        (a1 + a2 * t + a3 * t**2) + 2 * np.exp(x_**2 / 2)
    return sqrtpiover2 * out


@njit(signature0)
def log_erf_times_gauss_numba0(x):
    if x < 4:
        return 0.5 * x**2 + np.log(1 - erf_numba0(x / sqrt2)) + logsqrtpiover2
    else:
        return - np.log(x) + np.log(1 - 1 / x**2 + 3 / x**4)


@njit(signature1, parallel=True)
def log_erf_times_gauss_numba1(x):
    out = np.zeros(x.shape, dtype=curr_float)
    for i in prange(x.shape[0]):
        x_ = x[i]
        if x_ < 4:
            out[i] = 0.5 * x_**2 + \
                np.log(1 - erf_numba0(x_ / sqrt2)) + logsqrtpiover2
        else:
            out[i] = - np.log(x_) + np.log(1 - 1 / x_**2 + 3 / x_**4)
    return out


@njit("float64[:](float64[:])", parallel=True)
def erfinv_numba1(x):
    out = np.zeros(x.shape[0], dtype=np.float64)
    for i in prange(x.shape[0]):
        out[i] = ndtri_fn((x[i] + 1) / 2.0) / sqrt2_double
    return out


@njit("float64[:,:](float64[:,:])", parallel=True)
def erfinv_numba2(x):
    out = np.zeros((x.shape[0], x.shape[1]), dtype=np.float64)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            out[i, j] = ndtri_fn((x[i, j] + 1) / 2.0) / sqrt2_double
    return out


@njit("float64[:,:,:](float64[:,:,:])", parallel=True)
def erfinv_numba3(x):
    out = np.zeros((x.shape[0], x.shape[1], x.shape[2]), dtype=np.float64)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                out[i, j, k] = ndtri_fn((x[i, j, k] + 1) / 2.0) / sqrt2_double
    return out


@njit(parallel=True)
def sample_from_inputs_Bernoulli_numba2(I, fields, out):
    B, N = I.shape
    rng = np.random.rand(B, N)
    for b in prange(B):
        for n in prange(N):
            if rng[b, n] < 1 / (1 + np.exp(-I[b, n] - fields[n])):
                out[b, n] = 1.
            else:
                out[b, n] = 0.
    return out


@njit(parallel=True)
def sample_from_inputs_Bernoulli_numba3(I, fields, out):
    N_PT, B, N = I.shape
    rng = np.random.rand(N_PT, B, N)
    for n_pt in prange(N_PT):
        for b in prange(B):
            for n in prange(N):
                if rng[n_pt, b, n] < 1 / (1 + np.exp(-I[n_pt, b, n] - fields[n_pt, n])):
                    out[n_pt, b, n] = 1.
                else:
                    out[n_pt, b, n] = 0.
    return out


@njit(parallel=True)
def sample_from_inputs_Spin_numba2(I, fields, out):
    B, N = I.shape
    rng = np.random.rand(B, N)
    for b in prange(B):
        for n in prange(N):
            if rng[b, n] < 1 / (1 + np.exp(-2 * (I[b, n] + fields[n]))):
                out[b, n] = 1.
            else:
                out[b, n] = -1.
    return out


@njit(parallel=True)
def sample_from_inputs_Spin_numba3(I, fields, out):
    N_PT, B, N = I.shape
    rng = np.random.rand(N_PT, B, N)
    for n_pt in prange(N_PT):
        for b in prange(B):
            for n in prange(N):
                if rng[n_pt, b, n] < 1 / (1 + np.exp(-2 * (I[n_pt, b, n] + fields[n_pt, n]))):
                    out[n_pt, b, n] = 1.
                else:
                    out[n_pt, b, n] = -1.
    return out


@njit(parallel=True)
def sample_from_inputs_Potts_numba2(I, fields, out):
    B, N, n_c = I.shape
    rng = np.random.rand(B, N)
    cum_proba = np.empty(n_c, dtype=curr_float)
    for b in prange(B):
        for n in prange(N):
            rng_ = rng[b, n]
            cum_proba = I[b, n] + fields[n]
            cum_proba -= cum_proba.max()
            cum_proba[0] = np.exp(cum_proba[0])
            for c in range(1, n_c):
                cum_proba[c] = np.exp(cum_proba[c]) + cum_proba[c - 1]
            cum_proba /= cum_proba[-1]
            low = 0
            high = n_c
            while low < high:
                middle = (low + high) // 2
                if rng_ < cum_proba[middle]:
                    high = middle
                else:
                    low = middle + 1
            out[b, n] = high
    return out


@njit(parallel=True)
def sample_from_inputs_Potts_numba3(I, fields, out):
    N_PT, B, N, n_c = I.shape
    rng = np.random.rand(N_PT, B, N)
    cum_proba = np.empty(n_c, dtype=curr_float)
    for n_pt in prange(N_PT):
        for b in prange(B):
            for n in prange(N):
                rng_ = rng[n_pt, b, n]
                cum_proba = I[n_pt, b, n] + fields[n_pt, n]
                cum_proba -= cum_proba.max()
                cum_proba[0] = np.exp(cum_proba[0])
                for c in range(1, n_c):
                    cum_proba[c] = np.exp(cum_proba[c]) + cum_proba[c - 1]
                cum_proba /= cum_proba[-1]
                low = 0
                high = n_c
                while low < high:
                    middle = (low + high) // 2
                    if rng_ < cum_proba[middle]:
                        high = middle
                    else:
                        low = middle + 1
                out[n_pt, b, n] = high
    return out


@njit(parallel=True)
def sample_from_inputs_dReLU_numba2(I, gamma_plus, gamma_minus, theta_plus,
                                    theta_minus, out):
    B, N = I.shape
    rng = np.random.rand(2 * B, N)
    for b in prange(B):
        for n in prange(N):
            I_plus = (-I[b, n] + theta_plus[n]) / np.sqrt(gamma_plus[n])
            I_minus = (I[b, n] + theta_minus[n]) / np.sqrt(gamma_minus[n])
            etg_plus = erf_times_gauss_numba0(I_plus)
            etg_minus = erf_times_gauss_numba0(I_minus)
            p_plus = 1 / \
                (1 + (etg_minus /
                      np.sqrt(gamma_minus[n])) / (etg_plus / np.sqrt(gamma_plus[n])))
            if np.isnan(p_plus):
                if np.abs(I_plus) > np.abs(I_minus):
                    p_plus = 1
                else:
                    p_plus = 0
            # p_minus = 1 - p_plus
            is_pos = rng[b, n] < p_plus
            if is_pos:
                rmin = erf_numba0(I_plus * invsqrt2)
                rmax = 1
            else:
                rmin = -1
                rmax = erf_numba0(-I_minus * invsqrt2)

            out_ = ndtri_fn((rmin + (rmax - rmin) * rng[b + B, n] + 1) / 2.0)
            if is_pos:
                out_ = (out_ - I_plus) / np.sqrt(gamma_plus[n])
            else:
                out_ = (out_ + I_minus) / np.sqrt(gamma_minus[n])

            if np.isinf(out_) | np.isnan(out_) | (rmax - rmin < 1e-14):
                out_ = 0
            out[b, n] = out_
    return out


@njit(parallel=True)
def sample_from_inputs_dReLU_numba3(I, gamma_plus, gamma_minus, theta_plus,
                                    theta_minus, out):
    N_PT, B, N = I.shape
    rng = np.random.rand(2 * N_PT, B, N)
    for n_pt in prange(N_PT):
        for b in prange(B):
            for n in prange(N):
                I_plus = (-I[n_pt, b, n] + theta_plus[n_pt, n]) / \
                    np.sqrt(gamma_plus[n_pt, n])
                I_minus = (I[n_pt, b, n] + theta_minus[n_pt, n]) / \
                    np.sqrt(gamma_minus[n_pt, n])
                etg_plus = erf_times_gauss_numba0(I_plus)
                etg_minus = erf_times_gauss_numba0(I_minus)

                p_plus = 1 / (1 + (etg_minus / np.sqrt(gamma_minus[n_pt, n])) / (
                    etg_plus / np.sqrt(gamma_plus[n_pt, n])))
                if np.isnan(p_plus):
                    if np.abs(I_plus) > np.abs(I_minus):
                        p_plus = 1
                    else:
                        p_plus = 0
                # p_minus = 1 - p_plus
                is_pos = rng[n_pt, b, n] < p_plus
                if is_pos:
                    rmin = erf_numba0(I_plus * invsqrt2)
                    rmax = 1
                else:
                    rmin = -1
                    rmax = erf_numba0(-I_minus * invsqrt2)

                out_ = ndtri_fn(
                    (rmin + (rmax - rmin) * rng[N_PT + n_pt, b, n] + 1) / 2.0)
                if is_pos:
                    out_ = (out_ - I_plus) / np.sqrt(gamma_plus[n_pt, n])
                else:
                    out_ = (out_ + I_minus) / np.sqrt(gamma_minus[n_pt, n])

                if np.isinf(out_) | np.isnan(out_) | (rmax - rmin < 1e-14):
                    out_ = 0
                out[n_pt, b, n] = out_
    return out


@njit(parallel=True)
def cgf_from_inputs_dReLU_numba2(I, gamma_plus, gamma_minus, theta_plus,
                                 theta_minus):
    B, N = I.shape
    out = np.zeros(I.shape, dtype=curr_float)
    sqrt_gamma_plus = np.sqrt(gamma_plus)
    sqrt_gamma_minus = np.sqrt(gamma_minus)
    log_gamma_plus = np.log(gamma_plus)
    log_gamma_minus = np.log(gamma_minus)
    for b in prange(B):
        for n in prange(N):
            Z_plus = log_erf_times_gauss_numba0(
                (-I[b, n] + theta_plus[n]) / sqrt_gamma_plus[n]) - 0.5 * log_gamma_plus[n]
            Z_minus = log_erf_times_gauss_numba0(
                (I[b, n] + theta_minus[n]) / sqrt_gamma_minus[n]) - 0.5 * log_gamma_minus[n]
            if Z_plus > Z_minus:
                out[b, n] = Z_plus + np.log(1 + np.exp(Z_minus - Z_plus))
            else:
                out[b, n] = Z_minus + np.log(1 + np.exp(Z_plus - Z_minus))
    return out


@njit(parallel=True)
def cgf_from_inputs_dReLU_numba3(I, gamma_plus, gamma_minus, theta_plus,
                                 theta_minus):
    N_PT, B, N = I.shape
    out = np.zeros(I.shape, dtype=curr_float)
    sqrt_gamma_plus = np.sqrt(gamma_plus)
    sqrt_gamma_minus = np.sqrt(gamma_minus)
    log_gamma_plus = np.log(gamma_plus)
    log_gamma_minus = np.log(gamma_minus)
    for n_pt in prange(N_PT):
        for b in prange(B):
            for n in prange(N):
                Z_plus = log_erf_times_gauss_numba0(
                    (-I[n_pt, b, n] + theta_plus[n_pt, n]) / sqrt_gamma_plus[n_pt, n]) - 0.5 * log_gamma_plus[n_pt, n]
                Z_minus = log_erf_times_gauss_numba0(
                    (I[n_pt, b, n] + theta_minus[n_pt, n]) / sqrt_gamma_minus[n_pt, n]) - 0.5 * log_gamma_minus[n_pt, n]
                if Z_plus > Z_minus:
                    out[n_pt, b, n] = Z_plus + \
                        np.log(1 + np.exp(Z_minus - Z_plus))
                else:
                    out[n_pt, b, n] = Z_minus + \
                        np.log(1 + np.exp(Z_plus - Z_minus))
    return out


@njit(parallel=True)
def Bernoulli_Gibbs_free_C(x, fields_eff, B, N, fields0, couplings, beta):
    rng1 = np.random.randint(0, high=N, size=(B, N))
    rng2 = np.random.rand(B, N)

    for b in prange(B):
        for n in range(N):
            pos = rng1[b, n]
            previous = x[b, pos]
            if beta != 1:
                new = rng2[b, n] < 1 / (1 + np.exp(-beta *
                                                   fields_eff[b, pos] - (1 - beta) * fields0[pos]))
            else:
                new = rng2[b, n] < 1 / (1 + np.exp(-fields_eff[b, pos]))
            if new != previous:
                for n2 in range(N):
                    fields_eff[b, n2] += (new - previous) * couplings[n2, pos]
                x[b, pos] = new
    return x, fields_eff


@njit(parallel=True)
def Bernoulli_Gibbs_input_C(x, fields_eff, I, B, N, fields0, couplings, beta):
    rng1 = np.random.randint(0, high=N, size=(B, N))
    rng2 = np.random.rand(B, N)
    for b in prange(B):
        for n in range(N):
            pos = rng1[b, n]
            previous = x[b, pos]
            if beta != 1:
                new = rng2[b, n] < 1 / (1 + np.exp(-beta * fields_eff[b,
                                                                      pos] - (1 - beta) * fields0[pos] - I[b, pos]))
            else:
                new = rng2[b, n] < 1 / \
                    (1 + np.exp(-fields_eff[b, pos] - I[b, pos]))
            if new != previous:
                x[b, pos] = new
                for n2 in range(N):
                    fields_eff[b, n2] += (new - previous) * couplings[n2, pos]
    return x, fields_eff


@njit(parallel=True)
def Spin_Gibbs_free_C(x, fields_eff, B, N, fields0, couplings, beta):
    rng1 = np.random.randint(0, high=N, size=(B, N))
    rng2 = np.random.rand(B, N)
    for b in prange(B):
        for n in range(N):
            pos = rng1[b, n]
            previous = x[b, pos]
            if beta != 1:
                new = 2 * (rng2[b, n] < 1 / (1 + np.exp(-2 * (beta *
                                                              fields_eff[b, pos] + (1 - beta) * fields0[pos])))) - 1
            else:
                new = 2 * (rng2[b, n] < 1 /
                           (1 + np.exp(-2 * fields_eff[b, pos]))) - 1
            if new != previous:
                x[b, pos] = new
                for n2 in range(N):
                    fields_eff[b, n2] += 2 * new * couplings[n2, pos]

    return x, fields_eff


@njit(parallel=True)
def Spin_Gibbs_input_C(x, fields_eff, I, B, N, fields0, couplings, beta):
    rng1 = np.random.randint(0, high=N, size=(B, N))
    rng2 = np.random.rand(B, N)
    for b in prange(B):
        for n in range(N):
            pos = rng1[b, n]
            previous = x[b, pos]
            if beta != 1:
                new = 2 * (rng2[b, n] < 1 / (1 + np.exp(-2 * (beta * fields_eff[b,
                                                                                pos] + (1 - beta) * fields0[pos] + I[b, pos])))) - 1
            else:
                new = 2 * (rng2[b, n] < 1 / (1 + np.exp(-2 *
                                                        (fields_eff[b, pos] + I[b, pos])))) - 1
            if new != previous:
                x[b, pos] = new
                for n2 in range(N):
                    fields_eff[b, n2] += 2 * new * couplings[n2, pos]

    return x, fields_eff


@njit(parallel=True)
def Potts_Gibbs_free_C(x, fields_eff, B, N, n_c, fields0, couplings, beta):
    rng1 = np.random.randint(0, high=N, size=(B, N))
    rng2 = np.random.rand(B, N)
    for b in prange(B):
        for n in range(N):
            pos = rng1[b, n]
            previous = x[b, pos]
            cum_proba = np.cumsum(
                np.exp(beta * fields_eff[b, pos] + (1 - beta) * fields0[pos]))
            cum_proba /= cum_proba[-1]
            low = 0
            high = n_c
            while low < high:
                middle = (low + high) // 2
                if rng2[b, n] < cum_proba[middle]:
                    high = middle
                else:
                    low = middle + 1
            new = high
            if new != previous:
                x[b, pos] = new
                for n2 in range(N):
                    for c2 in range(n_c):
                        fields_eff[b, n2, c2] += couplings[n2, pos,
                                                           c2, new] - couplings[n2, pos, c2, previous]
    return x, fields_eff


@njit(parallel=True)
def Potts_Gibbs_input_C(x, fields_eff, I, B, N, n_c, fields0, couplings, beta):
    rng1 = np.random.randint(0, high=N, size=(B, N))
    rng2 = np.random.rand(B, N)
    for b in prange(B):
        for n in range(N):
            pos = rng1[b, n]
            previous = x[b, pos]
            cum_proba = np.cumsum(
                np.exp(beta * fields_eff[b, pos] + (1 - beta) * fields0[pos]))
            cum_proba /= cum_proba[-1]
            low = 0
            high = n_c
            while low < high:
                middle = (low + high) // 2
                if rng2[b, n] < cum_proba[middle]:
                    high = middle
                else:
                    low = middle + 1
            new = high
            if new != previous:
                x[b, pos] = new
                for n2 in range(N):
                    for c2 in range(n_c):
                        fields_eff[b, n2, c2] += couplings[n2, pos,
                                                           c2, new] - couplings[n2, pos, c2, previous]
    return x, fields_eff


if double_precision:
    signature = "Tuple((float64[:],float64[:],float64[:],float64[:],float64[:,:]))(float64[:,:],float64[:,:],float64[:],float64[:],float64[:],float64[:],float64[:] )"
else:
    signature = "Tuple((float32[:],float32[:],float32[:],float32[:],float32[:,:]))(float32[:,:],float32[:,:],float32[:],float32[:],float32[:],float32[:],float32[:] )"


@njit(signature, parallel=True)
def get_cross_derivatives_dReLU_numba(V, I,
                                      gamma, theta, eta, delta, weights):
    M = I.shape[1]
    N = V.shape[1]
    B = I.shape[0]
    sum_weights = weights.sum()
    mean_V = np.dot(weights, V) / sum_weights

    mean_e = np.zeros(M, dtype=curr_float)
    mean_e2 = np.zeros(M, dtype=curr_float)
    mean_v = np.zeros(M, dtype=curr_float)
    dmean_v_dgamma = np.zeros(M, dtype=curr_float)
    dmean_v_dtheta = np.zeros(M, dtype=curr_float)
    dmean_v_ddelta = np.zeros(M, dtype=curr_float)
    dmean_v_deta = np.zeros(M, dtype=curr_float)
    mean_eXde_dgamma = np.zeros(M, dtype=curr_float)
    mean_eXde_dtheta = np.zeros(M, dtype=curr_float)
    mean_eXde_ddelta = np.zeros(M, dtype=curr_float)
    mean_eXde_deta = np.zeros(M, dtype=curr_float)
    mean_de_dgamma = np.zeros(M, dtype=curr_float)
    mean_de_dtheta = np.zeros(M, dtype=curr_float)
    mean_de_ddelta = np.zeros(M, dtype=curr_float)
    mean_de_deta = np.zeros(M, dtype=curr_float)
    s1 = np.empty((B, M), dtype=curr_float)
    s2 = np.empty((B, M), dtype=curr_float)
    s3 = np.empty((B, M), dtype=curr_float)

    for m in prange(M):
        gamma_ = gamma[m]
        theta_ = theta[m]
        delta_ = delta[m]
        eta_ = eta[m]
        dI_plus_dI = -np.sqrt((1 + eta_) / gamma_)
        dI_minus_dI = np.sqrt((1 - eta_) / gamma_)
        dI_plus_ddelta = 1 / np.sqrt(gamma_)
        dI_minus_ddelta = 1 / np.sqrt(gamma_)
        d2I_plus_dgammadI = np.sqrt((1 + eta_) / gamma_**3) / 2
        d2I_plus_ddeltadI = 0
        d2I_plus_detadI = -1 / (2 * np.sqrt((1 + eta_) * gamma_))
        d2I_minus_dgammadI = -np.sqrt((1 - eta_) / gamma_**3) / 2
        d2I_minus_ddeltadI = 0
        d2I_minus_detadI = -1 / (2 * np.sqrt((1 - eta_) * gamma_))

        for b in prange(B):
            I_ = I[b, m]
            weights_ = weights[b]
            I_plus = (-np.sqrt(1 + eta_) * (I_ - theta_) +
                      delta_) / np.sqrt(gamma_)
            I_minus = (np.sqrt(1 - eta_) * (I_ - theta_) +
                       delta_) / np.sqrt(gamma_)
            etg_plus = erf_times_gauss_numba0(I_plus)
            etg_minus = erf_times_gauss_numba0(I_minus)

            Z = etg_plus * np.sqrt(1 + eta_) + etg_minus * np.sqrt(1 - eta_)

            p_plus = 1 / (1 + (etg_minus * np.sqrt(1 - eta_)) /
                          (etg_plus * np.sqrt(1 + eta_)))
            if np.isnan(p_plus):
                if np.abs(I_plus) > np.abs(I_minus):
                    p_plus = 1
                else:
                    p_plus = 0
            p_minus = 1 - p_plus

            e = (I_ - theta_) * (1 + eta_ * (p_plus - p_minus)) - delta_ * (np.sqrt(1 + eta_)
                                                                            * p_plus - np.sqrt(1 - eta_) * p_minus) + 2 * eta_ * np.sqrt(gamma_) / Z
            v = eta_ * (p_plus - p_minus) + p_plus * p_minus * ((np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_ / np.sqrt(gamma_) - 2 * eta_ * (I_ - theta_) / np.sqrt(gamma_)) * ((np.sqrt(1 + eta_) +
                                                                                                                                                                                     np.sqrt(1 - eta_)) * delta_ / np.sqrt(gamma_) - 2 * eta_ * (I_ - theta_) / np.sqrt(gamma_) - np.sqrt(1 + eta_) / etg_plus - np.sqrt(1 - eta_) / etg_minus) - 2 * eta_ * e / (np.sqrt(gamma_) * Z)

            dI_plus_dgamma = -1 / (2 * gamma_) * I_plus
            dI_minus_dgamma = -1 / (2 * gamma_) * I_minus
            dI_plus_deta = -1.0 / \
                (2 * np.sqrt(gamma_ * (1 + eta_))) * (I_ - theta_)
            dI_minus_deta = -1.0 / \
                (2 * np.sqrt(gamma_ * (1 - eta_))) * (I_ - theta_)

            dp_plus_dI = p_plus * p_minus * \
                ((I_plus - 1 / etg_plus) * dI_plus_dI -
                 (I_minus - 1 / etg_minus) * dI_minus_dI)
            dp_plus_ddelta = p_plus * p_minus * \
                ((I_plus - 1 / etg_plus) * dI_plus_ddelta -
                 (I_minus - 1 / etg_minus) * dI_minus_ddelta)
            dp_plus_dgamma = p_plus * p_minus * \
                ((I_plus - 1 / etg_plus) * dI_plus_dgamma -
                 (I_minus - 1 / etg_minus) * dI_minus_dgamma)
            dp_plus_deta = p_plus * p_minus * ((I_plus - 1 / etg_plus) * dI_plus_deta - (
                I_minus - 1 / etg_minus) * dI_minus_deta + 1 / (1 - eta_**2))

            d2p_plus_dI2 = -(p_plus - p_minus) * p_plus * p_minus * ((I_plus - 1 / etg_plus) * dI_plus_dI - (I_minus - 1 / etg_minus) * dI_minus_dI)**2 \
                + p_plus * p_minus * ((dI_plus_dI)**2 * (1 + (I_plus - 1 / etg_plus) / etg_plus) - (
                    dI_minus_dI)**2 * (1 + (I_minus - 1 / etg_minus) / etg_minus))

            d2p_plus_dgammadI = -(p_plus - p_minus) * ((I_plus - 1 / etg_plus) * dI_plus_dI - (I_minus - 1 / etg_minus) * dI_minus_dI) * (dp_plus_dgamma)\
                + p_plus * p_minus * ((dI_plus_dI * dI_plus_dgamma) * (1 + (I_plus - 1 / etg_plus) / etg_plus) - (dI_minus_dI * dI_minus_dgamma) * (1 + (I_minus - 1 / etg_minus) / etg_minus)
                                      + (d2I_plus_dgammadI) * (I_plus - 1 / etg_plus) - (d2I_minus_dgammadI) * (I_minus - 1 / etg_minus))

            d2p_plus_ddeltadI = -(p_plus - p_minus) * ((I_plus - 1 / etg_plus) * dI_plus_dI - (I_minus - 1 / etg_minus) * dI_minus_dI) * (dp_plus_ddelta)\
                + p_plus * p_minus * ((dI_plus_dI * dI_plus_ddelta) * (1 + (I_plus - 1 / etg_plus) / etg_plus) - (dI_minus_dI * dI_minus_ddelta) * (1 + (I_minus - 1 / etg_minus) / etg_minus)
                                      + (d2I_plus_ddeltadI) * (I_plus - 1 / etg_plus) - (d2I_minus_ddeltadI) * (I_minus - 1 / etg_minus))

            d2p_plus_detadI = -(p_plus - p_minus) * ((I_plus - 1 / etg_plus) * dI_plus_dI - (I_minus - 1 / etg_minus) * dI_minus_dI) * (dp_plus_deta)\
                + p_plus * p_minus * ((dI_plus_dI * dI_plus_deta) * (1 + (I_plus - 1 / etg_plus) / etg_plus) - (dI_minus_dI * dI_minus_deta) * (1 + (I_minus - 1 / etg_minus) / etg_minus)
                                      + (d2I_plus_detadI) * (I_plus - 1 / etg_plus) - (d2I_minus_detadI) * (I_minus - 1 / etg_minus))

            dlogZ_dI = (p_plus * (I_plus - 1 / etg_plus) * dI_plus_dI +
                        p_minus * (I_minus - 1 / etg_minus) * dI_minus_dI)
            dlogZ_ddelta = (p_plus * (I_plus - 1 / etg_plus) * dI_plus_ddelta +
                            p_minus * (I_minus - 1 / etg_minus) * dI_minus_ddelta)
            dlogZ_dgamma = (p_plus * (I_plus - 1 / etg_plus) * dI_plus_dgamma +
                            p_minus * (I_minus - 1 / etg_minus) * dI_minus_dgamma)
            dlogZ_deta = (p_plus * (I_plus - 1 / etg_plus) * dI_plus_deta + p_minus * (I_minus -
                                                                                       1 / etg_minus) * dI_minus_deta + (p_plus / (1 + eta_) - p_minus / (1 - eta_)) / 2)

            de_dI = (1 + v)
            de_dtheta = -de_dI
            de_dgamma = (2 * (I_ - theta_) * eta_ - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * \
                dp_plus_dgamma + eta_ / \
                (Z * np.sqrt(gamma_)) - 2 * eta_ * \
                np.sqrt(gamma_) / Z * dlogZ_dgamma
            de_ddelta = -(p_plus * np.sqrt(1 + eta_) - p_minus * np.sqrt(1 - eta_)) + (2 * (I_ - theta_) * eta_ - (
                np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * dp_plus_ddelta - 2 * eta_ * np.sqrt(gamma_) / Z * dlogZ_ddelta
            de_deta = (I_ - theta_) * (p_plus - p_minus) + (2 * (I_ - theta_) * eta_ - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * dp_plus_deta + 2 * np.sqrt(gamma_) / Z - 2 * eta_ * np.sqrt(gamma_) / Z * dlogZ_deta \
                - delta_ / 2 * (p_plus / np.sqrt(1 + eta_) +
                                p_minus / np.sqrt(1 - eta_))

            dv_dI = 4 * eta_ * dp_plus_dI\
                + (2 * (I_ - theta_) * eta_ - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * d2p_plus_dI2 \
                - 2 * eta_ / (np.sqrt(gamma_) * Z) * (de_dI - e * dlogZ_dI)

            dv_dtheta = -dv_dI

            dv_dgamma = eta_ * 2 * dp_plus_dgamma \
                + (2 * (I_ - theta_) * eta_ - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * d2p_plus_dgammadI \
                - 2 * eta_ / (Z * np.sqrt(gamma_)) * \
                (-e / (2 * gamma_) - e * dlogZ_dgamma + de_dgamma)

            dv_ddelta = 2 * eta_ * dp_plus_ddelta \
                - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * dp_plus_dI \
                + (2 * (I_ - theta_) * eta_ - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * d2p_plus_ddeltadI \
                - 2 * eta_ / (Z * np.sqrt(gamma_)) * \
                (- e * dlogZ_ddelta + de_ddelta)

            dv_deta = (p_plus - p_minus) \
                + 2 * eta_ * dp_plus_deta \
                + (2 * (I_ - theta_) - delta_ / 2 * (1 / np.sqrt(1 + eta_) - 1 / np.sqrt(1 - eta_))) * dp_plus_dI \
                + (2 * (I_ - theta_) * eta_ - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * d2p_plus_detadI \
                - 2 * 1 / (Z * np.sqrt(gamma_)) * \
                (e - e * eta_ * dlogZ_deta + eta_ * de_deta)

            mean_e[m] += e * weights_
            mean_e2[m] += e**2 * weights_
            mean_v[m] += v * weights_
            mean_de_dgamma[m] += de_dgamma * weights_
            mean_de_dtheta[m] += de_dtheta * weights_
            mean_de_ddelta[m] += de_ddelta * weights_
            mean_de_deta[m] += de_deta * weights_
            mean_eXde_dgamma[m] += e * de_dgamma * weights_
            mean_eXde_dtheta[m] += e * de_dtheta * weights_
            mean_eXde_ddelta[m] += e * de_ddelta * weights_
            mean_eXde_deta[m] += e * de_deta * weights_
            dmean_v_dgamma[m] += dv_dgamma * weights_
            dmean_v_dtheta[m] += dv_dtheta * weights_
            dmean_v_ddelta[m] += dv_ddelta * weights_
            dmean_v_deta[m] += dv_deta * weights_

            s1[b, m] = (dv_dI * weights_)
            s2[b, m] = (e * de_dI * weights_)
            s3[b, m] = (de_dI * weights_)

    dmean_v_dw = np.dot(s1.T, V)
    dvar_e_dw = np.dot(s2.T, V)
    tmp3 = np.dot(s3.T, V)

    mean_e /= sum_weights
    mean_e2 /= sum_weights
    mean_v /= sum_weights
    mean_de_dgamma /= sum_weights
    mean_de_dtheta /= sum_weights
    mean_de_ddelta /= sum_weights
    mean_de_deta /= sum_weights
    mean_eXde_dgamma /= sum_weights
    mean_eXde_dtheta /= sum_weights
    mean_eXde_ddelta /= sum_weights
    mean_eXde_deta /= sum_weights
    dmean_v_dgamma /= sum_weights
    dmean_v_dtheta /= sum_weights
    dmean_v_ddelta /= sum_weights
    dmean_v_deta /= sum_weights
    dmean_v_dw /= sum_weights
    dvar_e_dw /= sum_weights
    tmp3 /= sum_weights

    var_e = mean_e2 - mean_e**2
    dvar_e_dgamma = 2 * (mean_eXde_dgamma - mean_e * mean_de_dgamma)
    dvar_e_dtheta = 2 * (mean_eXde_dtheta - mean_e * mean_de_dtheta)
    dvar_e_ddelta = 2 * (mean_eXde_ddelta - mean_e * mean_de_ddelta)
    dvar_e_deta = 2 * (mean_eXde_deta - mean_e * mean_de_deta)
    dtheta_dw = mean_V

    tmp = np.sqrt((1 + mean_v)**2 + 4 * var_e)
    denominator = (tmp - dvar_e_dgamma -
                   dmean_v_dgamma * (1 + mean_v + tmp) / 2)
    dgamma_dtheta = (dvar_e_dtheta + dmean_v_dtheta *
                     (1 + mean_v + tmp) / 2) / denominator
    dgamma_ddelta = (dvar_e_ddelta + dmean_v_ddelta *
                     (1 + mean_v + tmp) / 2) / denominator
    dgamma_deta = (dvar_e_deta + dmean_v_deta *
                   (1 + mean_v + tmp) / 2) / denominator

    for m in prange(M):
        for n in prange(N):
            dvar_e_dw[m, n] -= mean_e[m] * tmp3[m, n]
    dvar_e_dw *= 2

    dgamma_dw = np.zeros((M, N), dtype=curr_float)
    for m in prange(M):
        for n in prange(N):
            dgamma_dw[m, n] = (dvar_e_dw[m, n] + dmean_v_dw[m, n] /
                               2 * (1 + mean_v[m] + tmp[m])) / denominator[m]

    return dtheta_dw, dgamma_dtheta, dgamma_ddelta, dgamma_deta, dgamma_dw


@njit(parallel=True)
def get_cross_derivatives_dReLU_Potts_numba(V, I,
                                            gamma, theta, eta, delta, weights, n_cv):
    M = I.shape[1]
    N = V.shape[1]
    B = I.shape[0]
    sum_weights = weights.sum()
    mean_V = weighted_average_C(V, weights, n_cv)

    mean_e = np.zeros(M, dtype=curr_float)
    mean_e2 = np.zeros(M, dtype=curr_float)
    mean_v = np.zeros(M, dtype=curr_float)
    dmean_v_dgamma = np.zeros(M, dtype=curr_float)
    dmean_v_dtheta = np.zeros(M, dtype=curr_float)
    dmean_v_ddelta = np.zeros(M, dtype=curr_float)
    dmean_v_deta = np.zeros(M, dtype=curr_float)
    mean_eXde_dgamma = np.zeros(M, dtype=curr_float)
    mean_eXde_dtheta = np.zeros(M, dtype=curr_float)
    mean_eXde_ddelta = np.zeros(M, dtype=curr_float)
    mean_eXde_deta = np.zeros(M, dtype=curr_float)
    mean_de_dgamma = np.zeros(M, dtype=curr_float)
    mean_de_dtheta = np.zeros(M, dtype=curr_float)
    mean_de_ddelta = np.zeros(M, dtype=curr_float)
    mean_de_deta = np.zeros(M, dtype=curr_float)
    s1 = np.empty((B, M), dtype=curr_float)
    s2 = np.empty((B, M), dtype=curr_float)
    s3 = np.empty((B, M), dtype=curr_float)

    for m in prange(M):
        gamma_ = gamma[m]
        theta_ = theta[m]
        delta_ = delta[m]
        eta_ = eta[m]
        dI_plus_dI = -np.sqrt((1 + eta_) / gamma_)
        dI_minus_dI = np.sqrt((1 - eta_) / gamma_)
        dI_plus_ddelta = 1 / np.sqrt(gamma_)
        dI_minus_ddelta = 1 / np.sqrt(gamma_)
        d2I_plus_dgammadI = np.sqrt((1 + eta_) / gamma_**3) / 2
        d2I_plus_ddeltadI = 0
        d2I_plus_detadI = -1 / (2 * np.sqrt((1 + eta_) * gamma_))
        d2I_minus_dgammadI = -np.sqrt((1 - eta_) / gamma_**3) / 2
        d2I_minus_ddeltadI = 0
        d2I_minus_detadI = -1 / (2 * np.sqrt((1 - eta_) * gamma_))

        for b in prange(B):
            I_ = I[b, m]
            weights_ = weights[b]
            I_plus = (-np.sqrt(1 + eta_) * (I_ - theta_) +
                      delta_) / np.sqrt(gamma_)
            I_minus = (np.sqrt(1 - eta_) * (I_ - theta_) +
                       delta_) / np.sqrt(gamma_)
            etg_plus = erf_times_gauss_numba0(I_plus)
            etg_minus = erf_times_gauss_numba0(I_minus)

            Z = etg_plus * np.sqrt(1 + eta_) + etg_minus * np.sqrt(1 - eta_)

            p_plus = 1 / (1 + (etg_minus * np.sqrt(1 - eta_)) /
                          (etg_plus * np.sqrt(1 + eta_)))
            if np.isnan(p_plus):
                if np.abs(I_plus) > np.abs(I_minus):
                    p_plus = 1
                else:
                    p_plus = 0
            p_minus = 1 - p_plus

            e = (I_ - theta_) * (1 + eta_ * (p_plus - p_minus)) - delta_ * (np.sqrt(1 + eta_)
                                                                            * p_plus - np.sqrt(1 - eta_) * p_minus) + 2 * eta_ * np.sqrt(gamma_) / Z
            v = eta_ * (p_plus - p_minus) + p_plus * p_minus * ((np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_ / np.sqrt(gamma_) - 2 * eta_ * (I_ - theta_) / np.sqrt(gamma_)) * ((np.sqrt(1 + eta_) +
                                                                                                                                                                                     np.sqrt(1 - eta_)) * delta_ / np.sqrt(gamma_) - 2 * eta_ * (I_ - theta_) / np.sqrt(gamma_) - np.sqrt(1 + eta_) / etg_plus - np.sqrt(1 - eta_) / etg_minus) - 2 * eta_ * e / (np.sqrt(gamma_) * Z)

            dI_plus_dgamma = -1 / (2 * gamma_) * I_plus
            dI_minus_dgamma = -1 / (2 * gamma_) * I_minus
            dI_plus_deta = -1.0 / \
                (2 * np.sqrt(gamma_ * (1 + eta_))) * (I_ - theta_)
            dI_minus_deta = -1.0 / \
                (2 * np.sqrt(gamma_ * (1 - eta_))) * (I_ - theta_)

            dp_plus_dI = p_plus * p_minus * \
                ((I_plus - 1 / etg_plus) * dI_plus_dI -
                 (I_minus - 1 / etg_minus) * dI_minus_dI)
            dp_plus_ddelta = p_plus * p_minus * \
                ((I_plus - 1 / etg_plus) * dI_plus_ddelta -
                 (I_minus - 1 / etg_minus) * dI_minus_ddelta)
            dp_plus_dgamma = p_plus * p_minus * \
                ((I_plus - 1 / etg_plus) * dI_plus_dgamma -
                 (I_minus - 1 / etg_minus) * dI_minus_dgamma)
            dp_plus_deta = p_plus * p_minus * ((I_plus - 1 / etg_plus) * dI_plus_deta - (
                I_minus - 1 / etg_minus) * dI_minus_deta + 1 / (1 - eta_**2))

            d2p_plus_dI2 = -(p_plus - p_minus) * p_plus * p_minus * ((I_plus - 1 / etg_plus) * dI_plus_dI - (I_minus - 1 / etg_minus) * dI_minus_dI)**2 \
                + p_plus * p_minus * ((dI_plus_dI)**2 * (1 + (I_plus - 1 / etg_plus) / etg_plus) - (
                    dI_minus_dI)**2 * (1 + (I_minus - 1 / etg_minus) / etg_minus))

            d2p_plus_dgammadI = -(p_plus - p_minus) * ((I_plus - 1 / etg_plus) * dI_plus_dI - (I_minus - 1 / etg_minus) * dI_minus_dI) * (dp_plus_dgamma)\
                + p_plus * p_minus * ((dI_plus_dI * dI_plus_dgamma) * (1 + (I_plus - 1 / etg_plus) / etg_plus) - (dI_minus_dI * dI_minus_dgamma) * (1 + (I_minus - 1 / etg_minus) / etg_minus)
                                      + (d2I_plus_dgammadI) * (I_plus - 1 / etg_plus) - (d2I_minus_dgammadI) * (I_minus - 1 / etg_minus))

            d2p_plus_ddeltadI = -(p_plus - p_minus) * ((I_plus - 1 / etg_plus) * dI_plus_dI - (I_minus - 1 / etg_minus) * dI_minus_dI) * (dp_plus_ddelta)\
                + p_plus * p_minus * ((dI_plus_dI * dI_plus_ddelta) * (1 + (I_plus - 1 / etg_plus) / etg_plus) - (dI_minus_dI * dI_minus_ddelta) * (1 + (I_minus - 1 / etg_minus) / etg_minus)
                                      + (d2I_plus_ddeltadI) * (I_plus - 1 / etg_plus) - (d2I_minus_ddeltadI) * (I_minus - 1 / etg_minus))

            d2p_plus_detadI = -(p_plus - p_minus) * ((I_plus - 1 / etg_plus) * dI_plus_dI - (I_minus - 1 / etg_minus) * dI_minus_dI) * (dp_plus_deta)\
                + p_plus * p_minus * ((dI_plus_dI * dI_plus_deta) * (1 + (I_plus - 1 / etg_plus) / etg_plus) - (dI_minus_dI * dI_minus_deta) * (1 + (I_minus - 1 / etg_minus) / etg_minus)
                                      + (d2I_plus_detadI) * (I_plus - 1 / etg_plus) - (d2I_minus_detadI) * (I_minus - 1 / etg_minus))

            dlogZ_dI = (p_plus * (I_plus - 1 / etg_plus) * dI_plus_dI +
                        p_minus * (I_minus - 1 / etg_minus) * dI_minus_dI)
            dlogZ_ddelta = (p_plus * (I_plus - 1 / etg_plus) * dI_plus_ddelta +
                            p_minus * (I_minus - 1 / etg_minus) * dI_minus_ddelta)
            dlogZ_dgamma = (p_plus * (I_plus - 1 / etg_plus) * dI_plus_dgamma +
                            p_minus * (I_minus - 1 / etg_minus) * dI_minus_dgamma)
            dlogZ_deta = (p_plus * (I_plus - 1 / etg_plus) * dI_plus_deta + p_minus * (I_minus -
                                                                                       1 / etg_minus) * dI_minus_deta + (p_plus / (1 + eta_) - p_minus / (1 - eta_)) / 2)

            de_dI = (1 + v)
            de_dtheta = -de_dI
            de_dgamma = (2 * (I_ - theta_) * eta_ - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * \
                dp_plus_dgamma + eta_ / \
                (Z * np.sqrt(gamma_)) - 2 * eta_ * \
                np.sqrt(gamma_) / Z * dlogZ_dgamma
            de_ddelta = -(p_plus * np.sqrt(1 + eta_) - p_minus * np.sqrt(1 - eta_)) + (2 * (I_ - theta_) * eta_ - (
                np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * dp_plus_ddelta - 2 * eta_ * np.sqrt(gamma_) / Z * dlogZ_ddelta
            de_deta = (I_ - theta_) * (p_plus - p_minus) + (2 * (I_ - theta_) * eta_ - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * dp_plus_deta + 2 * np.sqrt(gamma_) / Z - 2 * eta_ * np.sqrt(gamma_) / Z * dlogZ_deta \
                - delta_ / 2 * (p_plus / np.sqrt(1 + eta_) +
                                p_minus / np.sqrt(1 - eta_))

            dv_dI = 4 * eta_ * dp_plus_dI\
                + (2 * (I_ - theta_) * eta_ - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * d2p_plus_dI2 \
                - 2 * eta_ / (np.sqrt(gamma_) * Z) * (de_dI - e * dlogZ_dI)

            dv_dtheta = -dv_dI

            dv_dgamma = eta_ * 2 * dp_plus_dgamma \
                + (2 * (I_ - theta_) * eta_ - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * d2p_plus_dgammadI \
                - 2 * eta_ / (Z * np.sqrt(gamma_)) * \
                (-e / (2 * gamma_) - e * dlogZ_dgamma + de_dgamma)

            dv_ddelta = 2 * eta_ * dp_plus_ddelta \
                - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * dp_plus_dI \
                + (2 * (I_ - theta_) * eta_ - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * d2p_plus_ddeltadI \
                - 2 * eta_ / (Z * np.sqrt(gamma_)) * \
                (- e * dlogZ_ddelta + de_ddelta)

            dv_deta = (p_plus - p_minus) \
                + 2 * eta_ * dp_plus_deta \
                + (2 * (I_ - theta_) - delta_ / 2 * (1 / np.sqrt(1 + eta_) - 1 / np.sqrt(1 - eta_))) * dp_plus_dI \
                + (2 * (I_ - theta_) * eta_ - (np.sqrt(1 + eta_) + np.sqrt(1 - eta_)) * delta_) * d2p_plus_detadI \
                - 2 * 1 / (Z * np.sqrt(gamma_)) * \
                (e - e * eta_ * dlogZ_deta + eta_ * de_deta)

            mean_e[m] += e * weights_
            mean_e2[m] += e**2 * weights_
            mean_v[m] += v * weights_
            mean_de_dgamma[m] += de_dgamma * weights_
            mean_de_dtheta[m] += de_dtheta * weights_
            mean_de_ddelta[m] += de_ddelta * weights_
            mean_de_deta[m] += de_deta * weights_
            mean_eXde_dgamma[m] += e * de_dgamma * weights_
            mean_eXde_dtheta[m] += e * de_dtheta * weights_
            mean_eXde_ddelta[m] += e * de_ddelta * weights_
            mean_eXde_deta[m] += e * de_deta * weights_
            dmean_v_dgamma[m] += dv_dgamma * weights_
            dmean_v_dtheta[m] += dv_dtheta * weights_
            dmean_v_ddelta[m] += dv_ddelta * weights_
            dmean_v_deta[m] += dv_deta * weights_

            s1[b, m] = (dv_dI * weights_)
            s2[b, m] = (e * de_dI * weights_)
            s3[b, m] = (de_dI * weights_)

    mean_e /= sum_weights
    mean_e2 /= sum_weights
    mean_v /= sum_weights
    mean_de_dgamma /= sum_weights
    mean_de_dtheta /= sum_weights
    mean_de_ddelta /= sum_weights
    mean_de_deta /= sum_weights
    mean_eXde_dgamma /= sum_weights
    mean_eXde_dtheta /= sum_weights
    mean_eXde_ddelta /= sum_weights
    mean_eXde_deta /= sum_weights
    dmean_v_dgamma /= sum_weights
    dmean_v_dtheta /= sum_weights
    dmean_v_ddelta /= sum_weights
    dmean_v_deta /= sum_weights

    var_e = mean_e2 - mean_e**2
    dvar_e_dgamma = 2 * (mean_eXde_dgamma - mean_e * mean_de_dgamma)
    dvar_e_dtheta = 2 * (mean_eXde_dtheta - mean_e * mean_de_dtheta)
    dvar_e_ddelta = 2 * (mean_eXde_ddelta - mean_e * mean_de_ddelta)
    dvar_e_deta = 2 * (mean_eXde_deta - mean_e * mean_de_deta)
    dtheta_dw = mean_V

    tmp = np.sqrt((1 + mean_v)**2 + 4 * var_e)
    denominator = (tmp - dvar_e_dgamma -
                   dmean_v_dgamma * (1 + mean_v + tmp) / 2)

    dgamma_dtheta = (dvar_e_dtheta + dmean_v_dtheta *
                     (1 + mean_v + tmp) / 2) / denominator
    dgamma_ddelta = (dvar_e_ddelta + dmean_v_ddelta *
                     (1 + mean_v + tmp) / 2) / denominator
    dgamma_deta = (dvar_e_deta + dmean_v_deta *
                   (1 + mean_v + tmp) / 2) / denominator

    dmean_v_dw = average_product_FxP_C(s1, V, n_cv)
    dvar_e_dw = average_product_FxP_C(s2, V, n_cv)
    tmp3 = average_product_FxP_C(s3, V, n_cv)

    for m in prange(M):
        for n in prange(N):
            for c in prange(n_cv):
                dvar_e_dw[m, n, c] -= mean_e[m] * tmp3[m, n, c]
    dvar_e_dw *= 2

    dgamma_dw = np.zeros((M, N, n_cv), dtype=curr_float)
    for m in prange(M):
        for n in prange(N):
            for c in prange(n_cv):
                dgamma_dw[m, n, c] = (
                    dvar_e_dw[m, n, c] + dmean_v_dw[m, n, c] * (1 + mean_v[m] + tmp[m]) / 2) / denominator[m]
    return dtheta_dw, dgamma_dtheta, dgamma_ddelta, dgamma_deta, dgamma_dw
