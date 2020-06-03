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
# %% Useful functions
import numpy as np
import numbers
import numba_utilities as cy_utilities
from float_precision import double_precision, curr_float, curr_int
use_numba = True


def check_random_state(seed):
    if seed == None or seed == np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def gen_even_slices(n, n_packs, n_samples=None):
    start = 0
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            if n_samples != None:
                end = min(n_samples, end)
            yield slice(start, end, None)
            start = end


def check_nan(obj, what='', location=''):
    out = False
    if type(obj) == dict:
        for key, obj_ in obj.items():
            out = out | check_nan(
                obj_, what=what + ' ' + key, location=location)
    else:
        try:
            out = np.isnan(obj.max() )
        except:
            out = False
        if out:
            print('NAN in %s (%s). Breaking' % (what, location))
    return out


def logistic(x):
    return 1 / (1 + np.exp(-x))


def log_logistic(X, out=None):  # from SKLearn
    is_1d = X.ndim == 1

    if out is None:
        out = np.empty_like(X, dtype=curr_float)

    out = -np.logaddexp(0, -X)

    if is_1d:
        return np.squeeze(out)
    return out


def logsumexp(a, axis=None, b=None, keepdims=False):  # from scipy
    # This==a more elegant implementation, requiring NumPy >= 1.7.0
    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b != None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        out = np.log(np.sum(tmp, axis=axis, keepdims=keepdims))

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)

    out += a_max

    return out


if use_numba:
    def erf(x):
        return cy_utilities.erf_numba1(x.flatten()).reshape(x.shape)

    def erfinv(x):
        return cy_utilities.erfinv_numba1(x.astype(np.float64).flatten()).reshape(x.shape).astype(curr_float)

    def erf_times_gauss(x):
        return cy_utilities.erf_times_gauss_numba1(x.flatten()).reshape(x.shape)

    def log_erf_times_gauss(x):
        return cy_utilities.log_erf_times_gauss_numba1(x.flatten()).reshape(x.shape)

else:
    from scipy.special import erf, erfinv, erfcx

    if double_precision:
        maximum_etg = 6
    else:
        maximum_etg = 5

    # def erf_times_gauss(X):
    #     m = np.zeros(X.shape)
    #     tmp = X< maximum_etg
    #     m[tmp] = np.exp(0.5 * X[tmp]**2) * (1- erf(X[tmp]/np.sqrt(2))) * np.sqrt(np.pi/2);
    #     m[~tmp] = ( 1/X[~tmp] - 1/X[~tmp]**3 + 3/X[~tmp]**5);
    #     return m

    def erf_times_gauss(X):
        return erfcx(X / np.sqrt(2)) * np.sqrt(np.pi / 2)

    def log_erf_times_gauss(X):
        m = np.zeros(X.shape, dtype=curr_float)
        tmp = X < maximum_etg
        m[tmp] = (0.5 * X[tmp]**2 + np.log(1 -
                                           erf(X[tmp] / np.sqrt(2))) - np.log(2))
        m[~tmp] = (0.5 * np.log(2 / np.pi) - np.log(2) -
                   np.log(X[~tmp]) + np.log(1 - 1 / X[~tmp]**2 + 3 / X[~tmp]**4))
        return m


def softmax(X):
    X -= X.max(-1)[..., np.newaxis]
    np.exp(X, out=X)
    X /= (X.sum(-1))[..., np.newaxis]
    return X


def invert_softmax(mu, eps=1e-6, gauge='zerosum'):
    n_c = mu.shape[1]
    fields = np.log((1 - eps) * mu + eps / n_c)
    if gauge == 'zerosum':
        fields -= fields.sum(1)[:, np.newaxis] / n_c
    return fields


def cumulative_probabilities(X, maxi=1e9):
    X -= X.max(-1)[..., np.newaxis]
    np.exp(X, out=X)
    np.cumsum(X, axis=-1, out=X)
    X /= X[..., -1][..., np.newaxis]
    return X


def kronecker(X1, c1):
    l = []
    for color in range(c1):
        l.append(np.asarray(X1 == color, dtype=float))
    return np.rollaxis(np.array(l), 0, X1.ndim + 1)


def saturate(x, xmax):
    np.maximum(x, -xmax, out=x)
    np.minimum(x, xmax, out=x)
    return


def bound(x, xmin=None, xmax=None):
    if xmin is not None:
        np.maximum(x, xmin, out=x)
    if xmax is not None:
        np.minimum(x, xmax, out=x)
    return x


def average(X, c=1, weights=None):
    if weights is not None:
        if weights.dtype != curr_float:
            weights = weights.astype(curr_float)

    if (c == 1):
        if weights is None:
            return X.sum(0).astype(curr_float) / X.shape[0]
        else:
            if X.ndim == 1:
                return (X * weights).sum(0) / weights.sum()
            elif X.ndim == 2:
                return (X * weights[:, np.newaxis]).sum(0) / weights.sum()
            elif X.ndim == 3:
                return (X * weights[:, np.newaxis, np.newaxis]).sum(0) / weights.sum()
    else:
        if weights is None:
            return cy_utilities.average_C(X, c)
        else:
            return cy_utilities.weighted_average_C(X, weights, c)


def average_product(X1, X2, c1=1, c2=1, mean1=False, mean2=False, weights=None):
    if weights is not None:
        if weights.dtype != curr_float:
            weights = weights.astype(curr_float)

    if (c1 == 1) & (c2 == 1):
        if weights is None:
            return np.dot(X1.T, np.asarray(X2, dtype=curr_float)) / X1.shape[0]
        else:
            return (X1[:, :, np.newaxis] * X2[:, np.newaxis, :] * weights[:, np.newaxis, np.newaxis]).sum(0) / weights.sum()
    elif (c1 == 1) & (c2 != 1):
        # X2 in format [data,site,color]; each conditional mean for each color.
        if mean2:
            if weights is None:
                return np.tensordot(X1, X2, axes=([0], [0])) / X1.shape[0]
            else:
                return np.tensordot(X1 * weights[:, np.newaxis], X2, axes=([0], [0])) / weights.sum()
        else:
            if weights is None:
                return cy_utilities.average_product_FxP_C(np.asarray(X1, dtype=curr_float), X2, c2)
            else:
                return cy_utilities.average_product_FxP_C(np.asarray(X1 * weights[:, np.newaxis], dtype=curr_float), X2, c2) * weights.shape[0] / weights.sum()
    elif (c1 != 1) & (c2 == 1):
        # X1 in format [data,site,color]; each conditional mean for each color.
        if mean1:
            if weights is None:
                return np.swapaxes(np.tensordot(X1, X2, axes=([0], [0])), 1, 2) / X1.shape[0]
            else:
                return np.swapaxes(np.tensordot(X1, weights[:, np.newaxis, :] * X2, axes=([0], [0])), 1, 2) / weights.sum()
        else:
            if weights is None:
                return np.swapaxes(cy_utilities.average_product_FxP_C(np.asarray(X2, dtype=curr_float), X1, c1), 0, 1)
            else:
                return np.swapaxes(cy_utilities.average_product_FxP_C(np.asarray(X2 * weights[:, np.newaxis], dtype=curr_float), X1, c1), 0, 1) * weights.shape[0] / weights.sum()
    elif (c1 != 1) & (c2 != 1):
        if mean1 & mean2:
            if weights is None:
                return np.swapaxes(np.tensordot(X1, X2, axes=([0], [0])), 1, 2) / (X1.shape[0])
            else:
                return np.swapaxes(np.tensordot(X1 * weights[:, np.newaxis, np.newaxis], X2, axes=([0], [0])), 1, 2) / weights.sum()
        elif mean1 & (~mean2):
            out = np.zeros([X1.shape[1], X2.shape[1], c1, c2])
            for color2 in range(c2):
                if weights is None:
                    out[:, :, :, color2] = np.swapaxes(np.tensordot(
                        X1, (X2 == color2), axes=([0], [0])), 1, 2) / (X1.shape[0])
                else:
                    out[:, :, :, color2] = np.swapaxes(np.tensordot(
                        X1 * weights[:, np.newaxis, np.newaxis], (X2 == color2), axes=([0], [0])), 1, 2) / weights.sum()
            return out
        elif (~mean1) & mean2:
            out = np.zeros([X1.shape[1], X2.shape[1], c1, c2])
            for color1 in range(c1):
                if weights is None:
                    out[:, :, color1, :] = np.tensordot(
                        (X1 == color1), X2, axes=([0], [0])) / (X1.shape[0])
                else:
                    out[:, :, color1, :] = np.tensordot(
                        (X1 == color1), weights[:, np.newaxis, np.newaxis] * X2, axes=([0], [0])) / weights.sum()
            return out

        else:
            if weights is None:
                return cy_utilities.average_product_PxP_C(X1, X2, c1, c2)
            else:
                return cy_utilities.weighted_average_product_PxP_C(X1, X2, weights, c1, c2)


def covariance(X1, X2, c1=1, c2=1, mean1=False, mean2=False, weights=None):
    if mean1:
        mu1 = average(X1, weights=weights)
    else:
        mu1 = average(X1, c=c1, weights=weights)
    if mean2:
        mu2 = average(X2, weights=weights)
    else:
        mu2 = average(X2, c=c2, weights=weights)

    prod = average_product(X1, X2, c1=c1, c2=c2,
                           mean1=mean1, mean2=mean2, weights=weights)

    if (c1 > 1) & (c2 > 1):
        covariance = prod - mu1[:, np.newaxis, :,
                                np.newaxis] * mu2[np.newaxis, :, np.newaxis, :]
    elif (c1 > 1) & (c2 == 1):
        covariance = prod - mu1[:, np.newaxis, :] * \
            mu2[np.newaxis, :, np.newaxis]
    elif (c1 == 1) & (c2 > 1):
        covariance = prod - mu1[:, np.newaxis,
                                np.newaxis] * mu2[np.newaxis, :, :]
    else:
        covariance = prod - mu1[:, np.newaxis] * mu2[np.newaxis, :]
    return covariance


def bilinear_form(W, X1, X2, c1=1, c2=1):
    xshape = X1.shape
    X1, xdim = reshape_in(X1, xdim=1)
    X2, xdim = reshape_in(X2, xdim=1)
    if (c1 == 1) & (c2 == 1):
        out = np.sum(X1 * np.tensordot(X2, W, axes=(-1, 1)), -1)
    elif (c1 == 1) & (c2 > 1):
        out = np.sum(X1 * cy_utilities.compute_output_C(X2, W,
                                                        np.zeros(X1.shape, dtype=curr_float)), -1)
    elif (c1 > 1) & (c2 == 1):
        out = cy_utilities.dot_Potts2_C(
            X1.shape[1], c1, X1, np.tensordot(X2.astype(curr_float), W, (1, 1)))
    elif (c1 > 1) & (c2 > 1):
        out = cy_utilities.bilinear_form_Potts_C(X1,X2,W)
    return reshape_out(out, xshape, xdim=1)


def copy_config(config, N_PT=1, record_replica=False):
    if type(config) == tuple:
        if N_PT > 1:
            if record_replica:
                return config[0].copy()
            else:
                return config[0][0].copy()
        else:
            return config[0].copy()
    else:
        if N_PT > 1:
            if record_replica:
                return config.copy()
            else:
                return config[0].copy()
        else:
            return config.copy()


def make_all_discrete_configs(N, nature, c=1):
    iter_configurations = []
    if nature == 'Bernoulli':
        string = ','.join(['[0,1]' for _ in range(N)])
        exec('iter_configurations=itertools.product(%s)' % string)
    elif nature == 'Spin':
        string = ','.join(['[-1,1]' for _ in range(N)])
        exec('iter_configurations=itertools.product(%s)' % string)
    elif nature == 'Potts':
        liste_configs = '[' + ','.join([str(c) for c in range(c)]) + ']'
        string = ','.join([liste_configs for _ in range(N)])
        exec('iter_configurations=itertools.product(%s)' % string)
    else:
        print('no supported')
    configurations = np.array([config for config in iter_configurations])
    return configurations


def add_to_gradient(grad_x, grad_y, dy_dx):
    d1 = grad_x.ndim
    d2 = grad_y.ndim
    d3 = dy_dx.ndim
    case0 = (d1 == 1) & (d2 == 1) & (d3 == 1)  # e.g. y = gamma, x = eta.
    # e.g. y = fields Bernoulli, x = W.
    case1 = (d1 == 2) & (d2 == 1) & (d3 == 1)
    # e.g. y = fields Bernoulli, x = W. Potts.
    case2 = (d1 == 3) & (d2 == 1) & (d3 == 2)
    case3 = (d1 == 3) & (d2 == 2) & (d3 == 1)  # e.g. y = fields Potts, x = W.
    # e.g. y = fields Potts, x = W. Potts.
    case4 = (d1 == 4) & (d2 == 2) & (d3 == 2)

    if case0:
        grad_x += grad_y * dy_dx
    elif case1:
        grad_x += grad_y[:, np.newaxis] * dy_dx[np.newaxis, :]
    elif case2:
        grad_x += grad_y[:, np.newaxis, np.newaxis] * dy_dx[np.newaxis, :, :]
    elif case3:
        grad_x += grad_y[:, np.newaxis, :] * dy_dx[np.newaxis, :, np.newaxis]
    elif case4:
        grad_x += grad_y[:, np.newaxis, :, np.newaxis] * \
            dy_dx[np.newaxis, :, np.newaxis, :]
    return


def reshape_in(x, xdim=2):
    xshape = list(x.shape)
    ndims = len(xshape)
    if ndims == xdim:
        x = x[None]
    elif ndims > xdim + 1:
        x = x.reshape([np.prod(xshape[:-xdim])] + xshape[-xdim:])
    return x, xshape


def reshape_out(y, xshape, xdim=2):
    ndims = len(xshape)
    if ndims == xdim:
        return y[0]
    elif ndims > xdim + 1:
        return y.reshape(list(xshape[:-xdim]) + list(y.shape)[1:])
    else:
        return y


def get_permutation(N_PT, count):
    if N_PT == 2:
        return np.array([1, 0])
    else:
        permutation = np.arange(N_PT)
        if (N_PT % 2 == 0) & (count % 2 == 0):
            permutation -= 2 * (permutation % 2) - 1
        elif (N_PT % 2 == 0) & (count % 2 == 1):
            permutation[1:-1] += 2 * (permutation[1:-1] % 2) - 1
        elif (N_PT % 2 == 1) & (count % 2 == 0):
            permutation[:-1] -= 2 * (permutation[:-1] % 2) - 1
        else:
            permutation[1:] += 2 * (permutation[1:] % 2) - 1
        return permutation


def get_indices_swaps(N_PT, count):
    if N_PT == 2:
        return [0]
    else:
        return range(count % 2, N_PT - 1, 2)


# def get_permutation(N_PT, count):
#     permutation = np.arange(N_PT)
#     if (N_PT % 2 == 0) & (count % 2 == 0):
#         permutation -= 2 * (permutation % 2) - 1
#     elif (N_PT % 2 == 0) & (count % 2 == 1):
#         permutation[1:-1] += 2 * (permutation[1:-1] % 2) - 1
#     elif (N_PT % 2 == 1) & (count % 2 == 0):
#         permutation[:-1] -= 2 * (permutation[:-1] % 2) - 1
#     else:
#         permutation[1:] += 2 * (permutation[1:] % 2) - 1
#     return permutation
#
#
# def get_indices_swaps(N_PT, count):
#     return range(count % 2, N_PT - 1, 2)
