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
import numba_utilities as cy_utilities
import batch_norm_utils as batch_norm_utils
from utilities import check_random_state, logistic, softmax, invert_softmax, erf_times_gauss, log_erf_times_gauss, logsumexp, average, bilinear_form, average_product, covariance, add_to_gradient, saturate, erf, erfinv, reshape_in, reshape_out, bound

from float_precision import double_precision, curr_float, curr_int

# %% Layer class


def initLayer(nature='Bernoulli', **kwargs):
    exec('layer=%sLayer(**kwargs)' % nature)
    return locals()['layer']


class Layer():
    def __init__(self, N=100, nature='Bernoulli', position='visible', batch_norm=False, n_c=1, random_state=None):
        self.N = N
        self.nature = nature
        self.random_state = check_random_state(random_state)
        self.position = position
        self.n_c = n_c
        self.previous_beta = None
        if 'Potts' in self.nature:
            self.type = curr_int
        else:
            self.type = curr_float
        # Question: Why are Bernoulli and Spin of type curr_float and not curr_int?
        # Answer: in compute_output, np.dot, the repeated broadcasting from curr_int to curr_float takes a surprisingly long time for large arrays.

        self.batch_norm = batch_norm
        self.update0 = False
        self.target0 = None
        self.factors = []
        # Dictionary of the form params:(mini,maxi) for params that are bounded.
        self.params_constrained = {}

        if self.position == 'hidden':
            if self.n_c > 1:
                # For batch_norm.
                self.mu_I = np.zeros([N, n_c], dtype=curr_float)
            else:
                self.mu_I = np.zeros(N, dtype=curr_float)  # For batch_norm.

        # For batch_norm.
        if self.nature in ['Gaussian', 'ReLU', 'dReLU', 'ReLU+']:
            self.gamma_min = 0.05
            self.jump_max = 20
            self.gamma_drop_max = 0.75

    def get_input(self, I, I0=None, beta=1):
        if I is None:
            if self.n_c > 1:
                I = np.zeros([1, self.N, self.n_c], dtype=curr_float)
            else:
                I = np.zeros([1, self.N], dtype=curr_float)

        if type(beta) == np.ndarray:
            if self.n_c > 1:
                beta = beta[:, np.newaxis, np.newaxis, np.newaxis]
            else:
                beta = beta[:, np.newaxis, np.newaxis]
            beta_not_one = True
        else:
            beta_not_one = (beta != 1)

        if beta_not_one:
            I = I * beta
            if I0 is not None:
                I = I + (1 - beta) * I0
        if I.ndim == 1:
            # ensure that the config data is a batch, at least of just one vector
            I = I[np.newaxis, :]
        return I

    def get_params(self, beta=1):
        if type(beta) == np.ndarray:
            if self.n_c > 1:
                beta = beta[:, np.newaxis, np.newaxis, np.newaxis]
            else:
                beta = beta[:, np.newaxis, np.newaxis]
            beta_is_array = True
            beta_not_one = True
        else:
            beta_not_one = (beta != 1)
            beta_is_array = False

        for key in self.list_params:
            if beta_not_one & self.params_anneal[key]:
                tmp = beta * getattr(self, key) + \
                    (1 - beta) * getattr(self, key + '0')
            else:
                tmp = getattr(self, key)
            if self.params_newaxis[key] and not beta_is_array:
                tmp = tmp[np.newaxis, :]
            setattr(self, '_' + key, tmp)

    def compute_output(self, config, couplings, direction='up', out=None):
        case1 = (self.n_c > 1) & (couplings.ndim == 4)  # output layer is Potts
        case2 = (self.n_c > 1) & (couplings.ndim == 3)
        case3 = (self.n_c == 1) & (
            couplings.ndim == 3)  # output layer is Potts
        case4 = (self.n_c == 1) & (couplings.ndim == 2)

        if direction == 'up':
            N_output_layer = couplings.shape[0]
            if case1 | case3:
                n_c_output_layer = couplings.shape[2]
            else:
                n_c_output_layer = 1
        else:
            N_output_layer = couplings.shape[1]
            if case1 | case3:
                n_c_output_layer = couplings.shape[-1]
            else:
                n_c_output_layer = 1

        case1a = case1 & (N_output_layer == 1)
        case1b = case1 & (self.N == 1)
        case2a = case2 & (N_output_layer == 1)
        case2b = case2 & (self.N == 1)
        case3a = case3 & (N_output_layer == 1)
        case3b = case3 & (self.N == 1)
        case4a = case4 & (N_output_layer == 1)
        case4b = case4 & (self.N == 1)

        config, xshape = reshape_in(config, xdim=1)

        if case1 | case3:
            out_dim = list(xshape[:-1]) + [N_output_layer, n_c_output_layer]
            out_ndim = 2
        else:
            out_dim = list(xshape[:-1]) + [N_output_layer]
            out_ndim = 1
        if out is not None:
            if not list(out.shape) == out_dim:
                print('Mismatch dimensions %s, %s, reinitializating I' %
                      (out.shape, out_dim))
                out = np.zeros(out_dim, dtype=curr_float)
            else:
                out *= 0
        else:
            out = np.zeros(out_dim, dtype=curr_float)

        out, _ = reshape_in(out, xdim=out_ndim)

        if direction == 'up':
            if case1:
                if case1a:
                    out[:, 0, :] = cy_utilities.compute_output_C2(
                        config, couplings[0], out[:, 0, :])
                else:
                    out = cy_utilities.compute_output_Potts_C(
                        config, couplings, out)
            elif case2:
                out = cy_utilities.compute_output_C(config, couplings, out)
            elif case3:
                if case3a:
                    out[:, 0, :] = np.dot(
                        config, couplings[0], out=out[:, 0, :])
                else:
                    out = np.dot(config, couplings, out=out)
            else:
                out = np.dot(config, couplings.T, out=out)

        elif direction == 'down':
            if case1:
                if case1b:
                    out = cy_utilities.substitute_1C(
                        config[..., 0], couplings[0], out)
                else:
                    out = cy_utilities.compute_output_Potts_C2(
                        config, couplings, out)
            elif case2:
                if case2b:
                    out = cy_utilities.substitute_0C(
                        config[..., 0], couplings[0], out)
                else:
                    out = cy_utilities.compute_output_C2(
                        config, couplings, out)
            elif case3:
                out = np.tensordot(config, couplings, axes=(-1, 0))
            else:
                out = np.dot(config, couplings, out=out)
        return reshape_out(out, xshape, xdim=1)

    def logpartition(self, I, I0=None, beta=1):
        return self.cgf_from_inputs(I, I0=I0, beta=beta).sum(-1)

    def internal_gradients(self, data_pos, data_neg, data_0=None, weights=None, weights_neg=None, weights_0=None,
                           value='data', value_neg='data', value_0='input'):
        gradient = {}
        if value == 'moments':
            moments_pos = data_pos
        else:
            moments_pos = self.get_moments(
                data_pos,  value=value, weights=weights, beta=1)
        if value_neg == 'moments':
            moments_neg = data_neg
        else:
            moments_neg = self.get_moments(
                data_neg,  value=value_neg, weights=weights_neg, beta=1)
        if self.update0:
            if value_0 == 'moments':
                moments_0 = data_0
            elif value_0 == 'input':
                moments_0 = self.get_moments(
                    None, I0=data_0,  value=value_0, weights=weights_0, beta=0)
            else:
                moments_0 = self.get_moments(
                    data_0,  value=value_0, weights=weights_0, beta=0)

        if weights is not None:
            mean_weights = weights.mean()
        else:
            mean_weights = 1.

        if self.target0 == 'pos':
            self._target_moments0 = moments_pos
            self._mean_weight0 = mean_weights
        else:
            self._target_moments0 = moments_neg
            self._mean_weight0 = 1.

        for k, key in enumerate(self.list_params):
            gradient[key] = mean_weights * self.factors[k] * \
                (moments_pos[k] - moments_neg[k])
            if self.update0:
                gradient[key + '0'] = self._mean_weight0 * \
                    self.factors[k] * (self._target_moments0[k] - moments_0[k])
        return gradient

    def random_init_config(self, n_samples, N_PT=1):
        if not 'coupled' in self.nature:
            if self.n_c == 1:
                if N_PT > 1:
                    return self.sample_from_inputs(np.zeros([N_PT * n_samples, self.N], dtype=curr_float), beta=0).reshape([N_PT, n_samples, self.N])
                else:
                    return self.sample_from_inputs(np.zeros([n_samples, self.N], dtype=curr_float), beta=0).reshape([n_samples, self.N])
            else:
                if N_PT > 1:
                    return self.sample_from_inputs(np.zeros([N_PT * n_samples, self.N, self.n_c], dtype=curr_float), beta=0).reshape([N_PT, n_samples, self.N])
                else:
                    return self.sample_from_inputs(np.zeros([n_samples, self.N, self.n_c], dtype=curr_float), beta=0).reshape([n_samples, self.N])

        elif self.nature in ['Bernoulli_coupled', 'Spin_coupled']:
            if N_PT > 1:
                (x, fields_eff) = self.sample_from_inputs(
                    np.zeros([N_PT * n_samples, self.N], dtype=curr_float), beta=0)
                x = x.reshape([N_PT, n_samples, self.N])
                fields_eff = fields_eff.reshape([N_PT, n_samples, self.N])
            else:
                (x, fields_eff) = self.sample_from_inputs(
                    np.zeros([N_PT * n_samples, self.N], dtype=curr_float), beta=0)
            return (x, fields_eff)
        elif self.nature == 'Potts_coupled':
            if N_PT > 1:
                (x, fields_eff) = self.sample_from_inputs(
                    np.zeros([n_samples * N_PT, self.N, self.n_c], dtype=curr_float), beta=0)
                x = x.reshape([N_PT, n_samples, self.N])
                fields_eff = fields_eff.reshape(
                    [N_PT, n_samples, self.N, self.n_c])
            else:
                (x, fields_eff) = self.sample_from_inputs(
                    np.zeros([n_samples, self.N, self.n_c], dtype=curr_float), beta=0)
            return (x, fields_eff)

    def sample_and_energy_from_inputs(self, I, I0=None, beta=1, previous=(None, None), remove_init=False):
        if not 'coupled' in self.nature:
            config = self.sample_from_inputs(
                I, beta=beta, I0=I0, previous=previous)
            if remove_init:
                if I0 is not None:
                    I = I - I0
            else:
                I = self.get_input(I, I0=I0, beta=beta)

            energy = self.energy(config, remove_init=remove_init, beta=beta)
            if self.n_c == 1:
                energy -= (I * config).sum(-1)
            else:
                I, Idim = reshape_in(I, xdim=2)
                energy -= reshape_out(cy_utilities.dot_Potts2_C(
                    reshape_in(config, xdim=1)[0], I), Idim, xdim=2)
            return (config, energy)
        else:
            (x, fields_eff) = self.sample_from_inputs(
                I, I0=I0, beta=beta, previous=previous)
            if remove_init:
                f = 0.5 * (self.fields[np.newaxis] +
                           fields_eff) - self.fields0[np.newaxis]
                if I is not None:
                    f += I
                if I0 is not None:
                    f -= I0
            else:
                f = beta * fields_eff + (1 - beta) * self.fields0[np.newaxis]
                if I is not None:
                    f += beta * I
                if I0 is not None:
                    f += (1 - beta) * I0
            if self.nature == 'Potts_coupled':
                I, Idim = reshape_in(I, xdim=2)
                energy = - \
                    reshape_out(cy_utilities.dot_Potts2_C(
                        reshape_in(x, xdim=1)[0], f), Idim, xdim=2)
            else:
                energy = -np.sum(x * f, -1)
            return (x, fields_eff), energy

    # relevant when there is a change of variable.
    def recompute_params(self, which='regular'):
        return

    def ensure_constraints(self):
        for param_name, (mini, maxi) in self.params_constrained.items():
            param = getattr(self, param_name)
            bound(param, mini, maxi)
            if hasattr(self, param_name + '0'):
                param0 = getattr(self, param_name + '0')
                bound(param0, mini, maxi)

            # For non-linear interpolation param(beta) only. Ensuring the constraint for all beta is trickier...
            if hasattr(self, param_name + '1'):
                param1 = getattr(self, param_name + '1')
                if mini is not None:
                    lower_bound_minimum = np.minimum(param1[::2, :], 0).sum(
                        0) - np.abs(param1[1::2, :].sum(0))
                    ratio_mini = (mini - np.minimum(param, param0) -
                                  1e-10) / (lower_bound_minimum + 1e-10)
                else:
                    ratio_mini = np.ones(self.N, dtype=curr_float)
                if maxi is not None:
                    upper_bound_maximum = np.maximum(
                        param1[::2, :], 0).sum(-1) + np.abs(param1[1::2, :].sum(-1))
                    ratio_maxi = (maxi - np.maximum(param, param0) +
                                  1e-10) / (upper_bound_maximum + 1e-10)
                else:
                    ratio_maxi = np.ones(self.N, dtype=curr_float)

                pb_mini = ((ratio_mini < 1) & (ratio_mini > 0))
                pb_maxi = ((ratio_maxi < 1) & (ratio_maxi > 0))
                param1[:, pb_mini & pb_maxi] *= np.minimum(ratio_mini, ratio_maxi)[
                    pb_mini & pb_maxi][np.newaxis]
                param1[:, pb_mini & ~pb_maxi] *= ratio_mini[pb_mini &
                                                            ~pb_maxi][np.newaxis]
                param1[:, ~pb_mini & pb_maxi] *= ratio_maxi[~pb_mini &
                                                            pb_maxi][np.newaxis]
        return


class BernoulliLayer(Layer):
    def __init__(self, N=100, position='visible', random_state=None, batch_norm=False, zero_field=False, **kwargs):
        super(BernoulliLayer, self).__init__(N=N, nature='Bernoulli',
                                             position=position, batch_norm=batch_norm, n_c=1, random_state=random_state)
        self.zero_field = zero_field
        self.fields = np.zeros(self.N, dtype=curr_float)
        self.fields0 = np.zeros(self.N, dtype=curr_float)  # useful for PT.
        self.list_params = ['fields']
        self.params_anneal = {'fields': True}
        self.params_newaxis = {'fields': True}
        self.do_grad_updates = {'fields': ~
                                self.zero_field, 'fields0': (~self.zero_field)}
        self.do_grad_updates_batch_norm = self.do_grad_updates
        self.factors = [1]

    def mean_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        return logistic(I + self._fields)

    def var_from_inputs(self, I, I0=None, beta=1):
        mean = self.mean_from_inputs(I, I0=I0, beta=beta)
        return mean * (1 - mean)

    def cgf_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        return np.logaddexp(0, self._fields + I)

    def transform(self, I):
        self.get_params(beta=1)
        return (I + self._fields) > 0

    def sample_from_inputs(self, I, I0=None, beta=1, out=None, **kwargs):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        if out is None:
            out = np.empty_like(I)
        if type(beta) == np.ndarray:
            cy_utilities.sample_from_inputs_Bernoulli_numba3(
                I, self._fields[:, 0, :], out)
        else:
            cy_utilities.sample_from_inputs_Bernoulli_numba2(
                I, self._fields[0, :], out)
        return out

    def energy(self, config, remove_init=False, beta=1):
        if remove_init:
            return -np.dot(config, self.fields - self.fields0)
        else:
            self.get_params(beta=beta)
            return -(config * self._fields).sum(-1)

    def get_moments(self, data, I0=None, value='input', weights=None, beta=1):  # Bernoulli, Spin
        if value == 'input':
            mu = average(self.mean_from_inputs(
                data, I0=I0, beta=beta), weights=weights)
        else:
            mu = average(data, weights=weights)
        return (mu,)

    def init_params_from_data(self, X, eps=1e-6, value='data', weights=None):
        if X is None:
            self.fields = np.zeros(self.N, dtype=curr_float)
            self.fields0 = np.zeros(self.N, dtype=curr_float)
        else:
            if value == 'moments':
                moments = X
            else:
                moments = self.get_moments(X, value=value, weights=weights)
            self.fields = np.log((moments[0] + eps) / (1 - moments[0] + eps))
            self.fields0 = self.fields.copy()

    def batch_norm_update(self, mu_I, I, **kwargs):
        delta_mu_I = (mu_I - self.mu_I)
        self.mu_I = mu_I
        self.fields -= delta_mu_I

    def batch_norm_update_gradient(self, gradient_W, gradient_hlayer, V, I, mu, n_c, weights=None):
        add_to_gradient(gradient_W, gradient_hlayer['fields'], -mu)
        return


class SpinLayer(Layer):
    def __init__(self, N=100, position='visible', batch_norm=False, random_state=None, zero_field=False, **kwargs):
        super(SpinLayer, self).__init__(N=N, nature='Spin', position=position,
                                        batch_norm=batch_norm, n_c=1, random_state=random_state)
        self.zero_field = zero_field
        self.fields = np.zeros(self.N, dtype=curr_float)
        self.fields0 = np.zeros(self.N, dtype=curr_float)  # useful for PT.
        self.list_params = ['fields']
        self.params_anneal = {'fields': True}
        self.params_newaxis = {'fields': True}
        self.do_grad_updates = {'fields': ~
                                self.zero_field, 'fields0': (~self.zero_field)}
        self.do_grad_updates_batch_norm = self.do_grad_updates
        self.factors = [1]

    def mean_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        return np.tanh(I + self._fields)

    def var_from_inputs(self, I, I0=None, beta=1):
        return 1 - self.mean_from_inputs(I, I0=I0, beta=beta)**2

    def cgf_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        tmp = self._fields + I
        return np.logaddexp(-tmp, tmp)

    def transform(self, I):
        self.get_params(beta=1)
        return np.sign(I + self.fields)

    def sample_from_inputs(self, I, I0=None, beta=1, out=None, **kwargs):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        if out is None:
            out = np.empty_like(I)
        if type(beta) == np.ndarray:
            cy_utilities.sample_from_inputs_Spin_numba3(
                I, self._fields[:, 0, :], out)
        else:
            cy_utilities.sample_from_inputs_Spin_numba2(
                I, self._fields[0, :], out)
        return out

    def energy(self, config, remove_init=False, beta=1):
        if remove_init:
            return -np.dot(config, self.fields - self.fields0)
        else:
            self.get_params(beta=beta)
            return -(config * self._fields).sum(-1)

    def get_moments(self, data, I0=None, value='input', weights=None, beta=1):  # Bernoulli, Spin
        if value == 'input':
            mu = average(self.mean_from_inputs(
                data, I0=I0, beta=beta), weights=weights)
        else:
            mu = average(data, weights=weights)
        return (mu,)

    def init_params_from_data(self, X, eps=1e-6, value='data', weights=None):
        if X is None:
            self.fields = np.zeros(self.N, dtype=curr_float)
            self.fields0 = np.zeros(self.N, dtype=curr_float)
        else:
            if value == 'moments':
                moments = X
            else:
                moments = self.get_moments(X, value=value, weights=weights)
            self.fields = 0.5 * \
                np.log((1 + moments[0] + eps) / (1 - moments[0] + eps))
            self.fields0 = self.fields.copy()

    def batch_norm_update(self, mu_I, I, **kwargs):
        delta_mu_I = (mu_I - self.mu_I)
        self.mu_I = mu_I
        self.fields -= delta_mu_I

    def batch_norm_update_gradient(self, gradient_W, gradient_hlayer, V, I, mu, n_c, weights=None):
        add_to_gradient(gradient_W, gradient_hlayer['fields'], -mu)
        return


class PottsLayer(Layer):
    def __init__(self, N=100, position='visible', batch_norm=False, gauge='zerosum', n_c=2, random_state=None, zero_field=False, **kwargs):
        super(PottsLayer, self).__init__(N=N, nature='Potts', position=position,
                                         batch_norm=batch_norm, n_c=n_c, random_state=random_state)
        self.zero_field = zero_field
        self.fields = np.zeros([self.N, self.n_c], dtype=curr_float)
        self.fields0 = np.zeros([self.N, self.n_c], dtype=curr_float)
        self.gauge = gauge
        self.list_params = ['fields']
        self.params_anneal = {'fields': True}
        self.params_newaxis = {'fields': True}
        self.do_grad_updates = {'fields': ~
                                self.zero_field, 'fields0': (~self.zero_field)}
        self.do_grad_updates_batch_norm = self.do_grad_updates
        self.factors = [1]

    def mean_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        return softmax(I + self._fields)

    def var_from_inputs(self, I, I0=None, beta=1):
        self.get_params(beta=beta)
        mean = self.mean_from_inputs(I, I0=I0, beta=beta)
        return mean * (1 - mean)

    def cgf_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        return logsumexp(self._fields + I, -1)

    def transform(self, I):
        self.get_params(beta=1)
        return np.argmax(I + self._fields, axis=-1)

    def sample_from_inputs(self, I, I0=None, beta=1, out=None, **kwargs):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        if out is None:
            out = np.empty(I.shape[:-1], dtype=self.type)
        if type(beta) == np.ndarray:
            cy_utilities.sample_from_inputs_Potts_numba3(
                I, self._fields[:, 0, :, :], out)
        else:
            cy_utilities.sample_from_inputs_Potts_numba2(
                I, self._fields[0, :, :], out)
        return out

    def energy(self, config, remove_init=False, beta=1):
        beta_is_array = (type(beta) == np.ndarray)
        if remove_init:
            fields = self.fields - self.fields0
            config, dim = reshape_in(config, xdim=1)
            return reshape_out(-cy_utilities.dot_Potts_C(config, fields), dim, xdim=1)
        else:
            self.get_params(beta=beta)
            if beta_is_array:
                return -cy_utilities.dot_Potts3_C(config, self._fields[:, 0])
            else:
                return -cy_utilities.dot_Potts_C(config, self._fields[0])

    def get_moments(self, data, I0=None, value='input', weights=None, beta=1):  # Potts
        if value == 'input':
            mu = average(self.mean_from_inputs(
                data, I0=I0, beta=beta), weights=weights)
        elif value == 'mean':
            mu = average(data, weights=weights)
        else:
            mu = average(data, weights=weights, c=self.n_c)
        return (mu,)

    def init_params_from_data(self, X, eps=1e-6, value='data', weights=None):
        if X is None:
            self.fields = np.zeros(self.N, dtype=curr_float)
            self.fields0 = np.zeros(self.N, dtype=curr_float)
        else:
            if value == 'moments':
                moments = X
            else:
                moments = self.get_moments(X, value=value, weights=weights)
            self.fields = invert_softmax(moments[0], eps=eps, gauge=self.gauge)
            self.fields0 = self.fields.copy()

    def batch_norm_update(self, mu_I, I, **kwargs):
        delta_mu_I = (mu_I - self.mu_I)
        self.mu_I = mu_I
        self.fields -= delta_mu_I

    def batch_norm_update_gradient(self, gradient_W, gradient_hlayer, V, I, mu, n_c, weights=None):
        add_to_gradient(gradient_W, gradient_hlayer['fields'], -mu)
        return


class GaussianLayer(Layer):
    def __init__(self, N=100, position='visible', batch_norm=False, random_state=None, **kwargs):
        super(GaussianLayer, self).__init__(N=N, nature='Gaussian',
                                            position=position, batch_norm=batch_norm, n_c=1, random_state=random_state)
        self.gamma = np.ones(self.N, dtype=curr_float)
        self.gamma0 = np.ones(self.N, dtype=curr_float)
        self.theta = np.zeros(self.N, dtype=curr_float)
        self.theta0 = np.zeros(self.N, dtype=curr_float)
        self.list_params = ['gamma', 'theta']
        self.factors = [-0.5, -1]
        self.params_constrained = {'gamma': (self.gamma_min, None)}
        self.params_anneal = {'gamma': True, 'theta': True}
        self.params_newaxis = {'gamma': True, 'theta': True}
        if self.position == 'visible':
            self.do_grad_updates = {
                'gamma': True, 'theta': True, 'gamma0': True, 'theta0': True}
        else:
            self.do_grad_updates = {
                'gamma': False, 'theta': False, 'gamma0': True, 'theta0': True}
        self.do_grad_updates_batch_norm = {
            'gamma': False, 'theta': False, 'gamma0': True, 'theta0': True}

    def mean_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        return (I - self._theta) / self._gamma

    def mean2_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        return self.mean_from_inputs(I, I0=I0, beta=beta)**2 + 1 / self._gamma

    def var_from_inputs(self, I, I0=None, beta=1):
        self.get_params(beta=beta)
        return np.ones(I.shape) / self._gamma

    def cgf_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        return (0.5 * (I - self._theta)**2 / self._gamma) + 0.5 * np.log(2 * np.pi / self._gamma)

    def transform(self, I):
        return self.mean_from_inputs(I)

    def sample_from_inputs(self, I, I0=None, beta=1, out=None, **kwargs):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        if out is None:
            out = np.empty_like(I)
        out = (I - self._theta) / self._gamma + self.random_state.randn(*
                                                                        I.shape).astype(curr_float) / np.sqrt(self._gamma)
        return out

    def energy(self, config, remove_init=False, beta=1):
        if remove_init:
            return np.dot(config**2, self.gamma - self.gamma0) / 2 + np.dot(config, self.theta - self.theta0)
        else:
            self.get_params(beta=beta)
            return (config**2 * self._gamma).sum(-1) / 2 + (config * self._theta).sum(-1)

    def get_moments(self, data, I0=None, value='input', weights=None, beta=1):  # Gaussian, ReLU+
        if value == 'input':
            mu2 = average(self.mean2_from_inputs(
                data, I0=I0, beta=beta), weights=weights)
            mu = average(self.mean_from_inputs(
                data, I0=I0, beta=beta), weights=weights)
        else:
            mu2 = average(data**2, weights=weights)
            mu = average(data, weights=weights)
        return (mu2, mu)

    def init_params_from_data(self, X, eps=1e-6, value='data', weights=None):
        if X is None:
            self.gamma = np.ones(self.N, dtype=curr_float)
            self.gamma0 = np.ones(self.N, dtype=curr_float)
            self.theta = np.zeros(self.N, dtype=curr_float)
            self.theta0 = np.zeros(self.N, dtype=curr_float)
        else:
            if value == 'moments':
                moments = X
            else:
                moments = self.get_moments(X, value=value, weights=weights)
            mean = moments[1]
            var = moments[0] - moments[1]**2
            self.gamma = 1 / (var + eps)
            self.theta = - self.gamma * mean
            self.gamma0 = self.gamma.copy()
            self.theta0 = self.theta.copy()

    def batch_norm_update(self, mu_I, I, lr=1, weights=None):
        delta_mu_I = (mu_I - self.mu_I)
        self.mu_I = mu_I
        self.theta += delta_mu_I
        var_e = average(I**2, weights=weights) - average(I, weights=weights)**2
        new_gamma = (1 + np.sqrt(1 + 4 * var_e)) / 2
        self.gamma = np.maximum(self.gamma_min,
                                np.maximum(
                                    (1 - lr) * self.gamma + lr * new_gamma,
                                    self.gamma_drop_max * self.gamma
                                )
                                )

    def batch_norm_update_gradient(self, gradient_W, gradient_hlayer, V, I, mu, n_c, weights=None):
        WChat = covariance(I, V, weights=weights, c1=1, c2=n_c)
        var_e = average(I**2, weights=weights) - average(I, weights=weights)**2
        if n_c > 1:
            dgamma_dw = 2 / \
                np.sqrt(1 + 4 * var_e)[:, np.newaxis, np.newaxis] * WChat
        else:
            dgamma_dw = 2 / np.sqrt(1 + 4 * var_e)[:, np.newaxis] * WChat

        add_to_gradient(gradient_W, gradient_hlayer['theta'], mu)
        add_to_gradient(gradient_W, gradient_hlayer['gamma'], dgamma_dw)
        return


class ReLUplusLayer(Layer):
    def __init__(self, N=100, position='visible', batch_norm=False, random_state=None, **kwargs):
        super(ReLUplusLayer, self).__init__(N=N, nature='ReLUplus',
                                            position=position, batch_norm=batch_norm, n_c=1, random_state=random_state)
        self.gamma = np.ones(self.N, dtype=curr_float)
        self.gamma0 = np.ones(self.N, dtype=curr_float)
        self.theta = np.zeros(self.N, dtype=curr_float)
        self.theta0 = np.zeros(self.N, dtype=curr_float)
        self.list_params = ['gamma', 'theta']
        self.factors = [-0.5, -1.]
        self.params_anneal = {'gamma': True, 'theta': True}
        self.params_newaxis = {'gamma': True, 'theta': True}
        self.params_constrained = {'gamma': (0.05, None)}
        if self.position == 'visible':
            self.do_grad_updates = {
                'gamma': True, 'theta': True, 'gamma0': True, 'theta0': True}
        else:
            self.do_grad_updates = {
                'gamma': False, 'theta': True, 'gamma0': True, 'theta0': True}
        self.do_grad_updates_batch_norm = {
            'gamma': False, 'theta': True, 'gamma0': True, 'theta0': True}

    def mean_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        return (I - self._theta) / self._gamma + 1. / erf_times_gauss((-I + self._theta) / np.sqrt(self._gamma)) / np.sqrt(self._gamma)

    def mean2_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        return 1 / self._gamma * (1 + ((I - self._theta) / np.sqrt(self._gamma))**2 - ((-I + self._theta) / np.sqrt(self._gamma)) / erf_times_gauss((-I + self.theta) / np.sqrt(self._gamma)))

    def mean12_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        I_plus = (-I + self._theta) / np.sqrt(self._gamma)
        etg_plus = erf_times_gauss(I_plus)
        mean = (-I_plus + 1 / etg_plus) / np.sqrt(self._gamma)
        mean2 = 1 / self._gamma * (1 + I_plus**2 - I_plus / etg_plus)
        return mean, mean2

    def var_from_inputs(self, I, I0=None, beta=1):
        mean, mean2 = self.mean12_from_inputs(I, I0=I0, beta=beta)
        return mean2 - mean**2

    def cgf_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        return log_erf_times_gauss((-I + self._theta) / np.sqrt(self._gamma)) - 0.5 * np.log(self._gamma)

    def transform(self, I):
        self.get_params(beta=1)
        return np.maximum(I - self._theta, 0) / self._gamma

    def sample_from_inputs(self, I, I0=None, beta=1, out=None, **kwargs):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        I_plus = (-I + self._theta) / np.sqrt(self._gamma)
        rmin = erf(I_plus / np.sqrt(2))
        rmax = 1
        tmp = (rmax - rmin > 1e-14)
        h = (np.sqrt(2) * erfinv(rmin + (rmax - rmin) * self.random_state.random_sample(
            size=I.shape).astype(curr_float)) - I_plus) / np.sqrt(self._gamma)
        h[np.isinf(h) | np.isnan(h) | tmp] = 0
        return h

    def energy(self, config, remove_init=False, beta=1):
        if remove_init:
            return np.dot(config**2, self.gamma - self.gamma0) / 2 + np.dot(config, self.theta - self.theta0)
        else:
            self.get_params(beta=beta)
            return (config**2 * self._gamma).sum(-1) / 2 + (config * self._theta).sum(-1)

    def get_moments(self, data, I0=None, value='input', weights=None, average=False, beta=1):  # Gaussian, ReLU+
        if value == 'input':
            mu2 = average(self.mean2_from_inputs(
                data, I0=I0, beta=beta), weights=weights)
            mu = average(self.mean_from_inputs(
                data, I0=I0, beta=beta), weights=weights)
        else:
            mu2 = average(data**2, weights=weights)
            mu = average(data, weights=weights)
        return (mu2, mu)

    def init_params_from_data(self, X, eps=1e-6, value='data', weights=None):
        if X is None:
            self.gamma = np.ones(self.N, dtype=curr_float)
            self.gamma0 = np.ones(self.N, dtype=curr_float)
            self.theta = np.zeros(self.N, dtype=curr_float)
            self.theta0 = np.zeros(self.N, dtype=curr_float)
        else:
            if value == 'moments':
                moments = X
            else:
                moments = self.get_moments(X, value=value, weights=weights)
            mean = moments[1]
            var = moments[0] - moments[1]**2
            self.gamma = 1 / (var + eps)
            self.theta = - self.gamma * mean
            self.gamma0 = self.gamma.copy()
            self.theta0 = self.theta.copy()

    def batch_norm_update(self, mu_I, I, lr=1, weights=None):
        delta_mu_I = (mu_I - self.mu_I)
        self.mu_I = mu_I
        self.theta += delta_mu_I
        e = self.mean_from_inputs(I) * self.gamma[np.newaxis, :]
        v = (self.var_from_inputs(I) * self.gamma[np.newaxis, :] - 1)
        var_e = average(e**2, weights=weights) - average(e, weights=weights)**2
        mean_v = average(v, weights=weights)
        new_gamma = (1 + mean_v + np.sqrt((1 + mean_v)**2 + 4 * var_e)) / 2

        gamma_min = np.maximum(np.maximum(
            self.gamma_min,  # gamma cannot be too small
            self.gamma_drop_max * self.gamma)  # cannot drop too quickly.
        )
        self.gamma = np.maximum(
            (1 - lr) * self.gamma + lr * new_gamma, gamma_min)

    def batch_norm_update_gradient(self, gradient_W, gradient_hlayer, V, I, mu, n_c, weights=None):
        dtheta_dw, dgamma_dtheta, dgamma_dw = batch_norm_utils.get_cross_derivatives_ReLU_plus(
            V, I, self, n_c, weights=weights)
        add_to_gradient(gradient_hlayer['theta'],
                        gradient_hlayer['gamma'], dgamma_dtheta)
        add_to_gradient(gradient_W, gradient_hlayer['theta'], dtheta_dw)
        add_to_gradient(gradient_W, gradient_hlayer['gamma'], dgamma_dw)
        return


class ReLULayer(Layer):
    def __init__(self, N=100, position='visible', batch_norm=False, random_state=None, **kwargs):
        super(ReLULayer, self).__init__(N=N, nature='ReLU', position=position,
                                        batch_norm=batch_norm, n_c=1, random_state=random_state)
        self.gamma = np.ones(self.N, dtype=curr_float)
        self.gamma0 = np.ones(self.N, dtype=curr_float)
        self.theta_plus = np.zeros(self.N, dtype=curr_float)
        self.theta_plus0 = np.zeros(self.N, dtype=curr_float)
        self.theta_minus = np.zeros(self.N, dtype=curr_float)
        self.theta_minus0 = np.zeros(self.N, dtype=curr_float)

        # batch norm parametrization.
        self.theta = np.zeros(self.N, dtype=curr_float)
        # batch norm parametrization.
        self.delta = np.zeros(self.N, dtype=curr_float)

        self.list_params = ['gamma', 'theta_plus', 'theta_minus']
        self.factors = [-0.5, -1., -1.]
        self.params_anneal = {'gamma': True,
                              'theta_plus': True, 'theta_minus': True}
        self.params_newaxis = {'gamma': True,
                               'theta_plus': True, 'theta_minus': True}
        self.params_constrained = {'gamma': (self.gamma_min, None)}
        if self.position == 'visible':
            self.do_grad_updates = {'gamma': True, 'theta_plus': True, 'theta_minus': True,
                                    'theta': False, 'delta': False,
                                    'gamma0': False, 'theta_plus0': False, 'theta_minus0': False}
        else:
            self.do_grad_updates = {'gamma': False, 'theta_plus': True, 'theta_minus': True,
                                    'theta': False, 'delta': False,
                                    'gamma0': True, 'theta_plus0': True, 'theta_minus0': True}

        self.do_grad_updates_batch_norm = {'gamma': False, 'theta': True, 'delta': True,
                                           'theta_plus': False, 'theta_minus': False, 'gamma0': False, 'theta_plus0': True, 'theta_minus0': True}

    def mean_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        etg_plus = erf_times_gauss(
            (-I + self._theta_plus) / np.sqrt(self._gamma))
        etg_minus = erf_times_gauss(
            (I + self._theta_minus) / np.sqrt(self._gamma))
        p_plus = 1 / (1 + etg_minus / etg_plus)
        p_minus = 1 - p_plus
        mean_neg = (I + self._theta_minus) / self._gamma - \
            1 / etg_minus / np.sqrt(self._gamma)
        mean_pos = (I - self._theta_plus) / self._gamma + \
            1 / etg_plus / np.sqrt(self._gamma)
        return mean_pos * p_plus + mean_neg * p_minus

    def mean2_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        etg_plus = erf_times_gauss(
            (-I + self._theta_plus) / np.sqrt(self._gamma))
        etg_minus = erf_times_gauss(
            (I + self._theta_minus) / np.sqrt(self._gamma))
        p_plus = 1 / (1 + etg_minus / etg_plus)
        p_minus = 1 - p_plus
        mean2_pos = 1 / self._gamma * (1 + ((I - self._theta_plus) / np.sqrt(
            self._gamma))**2 - ((-I + self._theta_plus) / np.sqrt(self._gamma)) / etg_plus)
        mean2_neg = 1 / self._gamma * (1 + ((I + self._theta_minus) / np.sqrt(
            self._gamma))**2 - ((I + self._theta_minus) / np.sqrt(self._gamma)) / etg_minus)
        return (p_plus * mean2_pos + p_minus * mean2_neg)

    def mean_pm_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        etg_plus = erf_times_gauss(
            (-I + self._theta_plus) / np.sqrt(self._gamma))
        etg_minus = erf_times_gauss(
            (I + self._theta_minus) / np.sqrt(self._gamma))
        p_plus = 1 / (1 + etg_minus / etg_plus)
        p_minus = 1 - p_plus
        mean_neg = (I + self._theta_minus) / self._gamma - \
            1 / etg_minus / np.sqrt(self._gamma)
        mean_pos = (I - self._theta_plus) / self._gamma + \
            1 / etg_plus / np.sqrt(self._gamma)
        return (mean_pos * p_plus, mean_neg * p_minus)

    def mean12_pm_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        etg_plus = erf_times_gauss(
            (-I + self._theta_plus) / np.sqrt(self._gamma))
        etg_minus = erf_times_gauss(
            (I + self._theta_minus) / np.sqrt(self._gamma))
        p_plus = 1 / (1 + etg_minus / etg_plus)
        p_minus = 1 - p_plus
        mean_neg = (I + self._theta_minus) / self._gamma - \
            1 / etg_minus / np.sqrt(self._gamma)
        mean_pos = (I - self._theta_plus) / self._gamma + \
            1 / etg_plus / np.sqrt(self._gamma)
        mean2_pos = 1 / self._gamma * (1 + ((I - self._theta_plus) / np.sqrt(
            self._gamma))**2 - ((-I + self._theta_plus) / np.sqrt(self._gamma)) / etg_plus)
        mean2_neg = 1 / self._gamma * (1 + ((I + self._theta_minus) / np.sqrt(
            self._gamma))**2 - ((I + self._theta_minus) / np.sqrt(self._gamma)) / etg_minus)
        return (mean_pos * p_plus, mean_neg * p_minus, mean2_pos * p_plus, mean2_neg * p_minus)

    def var_from_inputs(self, I, I0=None, beta=1):
        (mu_pos, mu_neg, mu2_pos, mu2_neg) = self.mean12_pm_from_inputs(
            I, I0=I0, beta=beta)
        return (mu2_pos + mu2_neg) - (mu_pos + mu_neg)**2

    def cgf_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        return np.logaddexp(log_erf_times_gauss((-I + self._theta_plus) / np.sqrt(self._gamma)), log_erf_times_gauss((I + self._theta_minus) / np.sqrt(self._gamma))) - 0.5 * np.log(self._gamma)

    def transform(self, I):
        self.get_params(beta=1)
        return 1 / self._gamma * ((I + self._theta_minus) * (I <= np.minimum(-self._theta_minus, (self._theta_plus - self._theta_minus) / 2)) + (I - self._theta_plus) * (I >= np.maximum(self._theta_plus, (self._theta_plus - self._theta_minus) / 2)))

    def sample_from_inputs(self, I, I0=None, beta=1, out=None, **kwargs):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        I_plus = (-I + self._theta_plus) / np.sqrt(self._gamma)
        I_minus = (I + self._theta_minus) / np.sqrt(self._gamma)

        etg_plus = erf_times_gauss(I_plus)
        etg_minus = erf_times_gauss(I_minus)

        p_plus = 1 / (1 + etg_minus / etg_plus)
        nans = np.isnan(p_plus)
        p_plus[nans] = 1.0 * (np.abs(I_plus[nans]) > np.abs(I_minus[nans]))
        p_minus = 1 - p_plus

        is_pos = self.random_state.random_sample(size=I.shape) < p_plus
        rmax = np.zeros(I.shape, dtype=curr_float)
        rmin = np.zeros(I.shape, dtype=curr_float)
        rmin[is_pos] = erf(I_plus[is_pos] / np.sqrt(2))
        rmax[is_pos] = 1
        rmin[~is_pos] = -1
        rmax[~is_pos] = erf(-I_minus[~is_pos] / np.sqrt(2))

        h = np.zeros(I.shape, dtype=curr_float)
        tmp = (rmax - rmin > 1e-14)
        h = np.sqrt(2) * erfinv(rmin + (rmax - rmin) *
                                self.random_state.random_sample(size=h.shape).astype(curr_float))
        h[is_pos] -= I_plus[is_pos]
        h[~is_pos] += I_minus[~is_pos]
        h /= np.sqrt(self._gamma)
        h[np.isinf(h) | np.isnan(h) | ~tmp] = 0
        return h

    def energy(self, config, remove_init=False, beta=1):
        if remove_init:
            return np.dot(config**2, self.gamma - self.gamma0) / 2 + np.dot(np.maximum(config, 0), self.theta_plus - self.theta_plus0) + np.dot(np.maximum(-config, 0), self.theta_minus - self.theta_minus0)
        else:
            self.get_params(beta=beta)
            return (config**2 * self._gamma).sum(-1) / 2 + (np.maximum(config, 0) * self._theta_plus).sum(-1) + (np.maximum(-config, 0) * self._theta_minus).sum(-1)

    # ReLU.
    def get_moments(self, data, I0=None, value='input', weights=None, average=False, beta=1):
        if value == 'input':
            mu_pos, mu_neg, mu2_pos, mu2_neg = self.mean12_pm_from_inputs(
                data, I0=I0, beta=beta)
            mu_pos = average(mu_pos, weights=weights)
            mu_neg = average(mu_pos, weights=weights)
            mu2 = average(mu2_pos + mu2_neg, weights=weights)
        else:
            mu_pos = average(np.maximum(data, 0), weights=weights)
            mu_neg = average(np.minimum(data, 0), weights=weights)
            mu2 = average(data**2, weights=weights)
        return (mu2, mu_pos, mu_neg)

    def internal_gradients(self, **kwargs):  # ReLU
        gradients = super(ReLULayer, self).internal_gradients(**kwargs)
        if self.position == 'hidden':
            gradients['gamma'] = gradients['gamma_plus'] + \
                gradients['gamma_minus']
            gradients['theta'] = gradients['theta_plus'] - \
                gradients['theta_minus']
            gradients['delta'] = gradients['theta_plus'] + \
                gradients['theta_minus']
        return gradients

    def init_params_from_data(self, X, eps=1e-6, value='data', weights=None):
        if X is None:
            self.gamma = np.ones(self.N, dtype=curr_float)
            self.gamma0 = np.ones(self.N, dtype=curr_float)
            self.theta_plus = np.zeros(self.N, dtype=curr_float)
            self.theta_plus0 = np.zeros(self.N, dtype=curr_float)
            self.theta_minus = np.zeros(self.N, dtype=curr_float)
            self.theta_minus0 = np.zeros(self.N, dtype=curr_float)
            # batch norm parametrization.
            self.theta = np.zeros(self.N, dtype=curr_float)
            # batch norm parametrization.
            self.delta = np.zeros(self.N, dtype=curr_float)
        else:
            if value == 'moments':
                moments = X
            else:
                moments = self.get_moments(X, value=value, weights=weights)
            mean = moments[1]
            var = moments[0] - moments[1]**2
            self.gamma = 1 / (var + eps)
            self.theta_plus = - self.gamma * mean
            self.theta_minus = - self.theta_plus
            self.gamma0 = self.gamma.copy()
            self.theta_plus0 = self.theta_plus.copy()
            self.theta_minus0 = self.theta_minus.copy()

    def batch_norm_update(self, mu_I, I, lr=1, weights=None):
        delta_mu_I = (mu_I - self.mu_I)
        self.mu_I = mu_I
        self.theta += delta_mu_I
        self.theta_plus += delta_mu_I
        self.theta_minus -= delta_mu_I
        e = self.mean_from_inputs(I) * self.gamma[np.newaxis, :]
        v = (self.var_from_inputs(I) * self.gamma[np.newaxis, :] - 1)
        var_e = average(e**2, weights=weights) - average(e, weights=weights)**2
        mean_v = average(v, weights=weights)
        new_gamma = (1 + mean_v + np.sqrt((1 + mean_v)**2 + 4 * var_e)) / 2
        gamma_min = np.maximum(np.maximum(
            self.gamma_min,  # gamma cannot be too small
            self.gamma_drop_max * self.gamma)  # cannot drop too quickly.
        )
        self.gamma = np.maximum(
            (1 - lr) * self.gamma + lr * new_gamma, gamma_min)

    def batch_norm_update_gradient(self, gradient_W, gradient_hlayer, V, I, mu, n_c, weights=None):
        dtheta_dw, dgamma_dtheta, dgamma_ddelta, dgamma_dw = batch_norm_utils.get_cross_derivatives_ReLU(
            V, I, self, n_c, weights=weights)
        add_to_gradient(gradient_hlayer['theta'],
                        gradient_hlayer['gamma'], dgamma_dtheta)
        add_to_gradient(gradient_hlayer['delta'],
                        gradient_hlayer['delta'], dgamma_ddelta)
        add_to_gradient(gradient_W, gradient_hlayer['theta'], dtheta_dw)
        add_to_gradient(gradient_W, gradient_hlayer['gamma'], dgamma_dw)
        return

    def recompute_params(self, which='regular'):
        if which == 'regular':
            self.theta_plus = self.delta + self.theta
            self.theta_minus = self.delta - self.theta
        else:
            self.delta = (self.theta_plus + self.theta_minus) / 2
            self.theta = (self.theta_plus - self.theta_minus) / 2


class dReLULayer(Layer):
    def __init__(self, N=100, position='visible', batch_norm=False, random_state=None, **kwargs):
        super(dReLULayer, self).__init__(N=N, nature='dReLU', position=position,
                                         batch_norm=batch_norm, n_c=1, random_state=random_state)
        self.gamma_plus = np.ones(self.N, dtype=curr_float)
        self.gamma_plus0 = np.ones(self.N, dtype=curr_float)
        self.gamma_minus = np.ones(self.N, dtype=curr_float)
        self.gamma_minus0 = np.ones(self.N, dtype=curr_float)
        self.theta_plus = np.zeros(self.N, dtype=curr_float)
        self.theta_plus0 = np.zeros(self.N, dtype=curr_float)
        self.theta_minus = np.zeros(self.N, dtype=curr_float)
        self.theta_minus0 = np.zeros(self.N, dtype=curr_float)

        # batch norm parametrization.
        self.gamma = np.ones(self.N, dtype=curr_float)
        # batch norm parametrization.
        self.theta = np.zeros(self.N, dtype=curr_float)
        # batch norm parametrization.
        self.delta = np.zeros(self.N, dtype=curr_float)
        # batch norm parametrization.
        self.eta = np.zeros(self.N, dtype=curr_float)

        self.list_params = ['gamma_plus',
                            'gamma_minus', 'theta_plus', 'theta_minus']
        self.factors = [-0.5, -0.5, -1, +1]
        self.params_anneal = {
            'gamma_plus': True, 'gamma_minus': True, 'theta_plus': True, 'theta_minus': True}
        self.params_newaxis = {
            'gamma_plus': True, 'gamma_minus': True, 'theta_plus': True, 'theta_minus': True}
        self.params_constrained = {'gamma': (self.gamma_min, None), 'gamma_plus': (
            self.gamma_min, None), 'gamma_minus': (self.gamma_min, None)}
        if self.position == 'visible':
            self.do_grad_updates = {'gamma_plus': True, 'gamma_minus': True, 'theta_plus': True, 'theta_minus': True,
                                    'theta': False, 'delta': False, 'eta': False, 'gamma': False,
                                    'gamma_plus0': True, 'gamma_minus0': True, 'theta_plus0': True, 'theta_minus0': True}
        else:
            self.do_grad_updates = {'gamma_plus': False, 'gamma_minus': False, 'theta_plus': False, 'theta_minus': False,
                                    'theta': True, 'delta': True, 'eta': True, 'gamma': False,
                                    'gamma_plus0': True, 'gamma_minus0': True, 'theta_plus0': True, 'theta_minus0': True}

        self.do_grad_updates_batch_norm = {'gamma_plus': False, 'gamma_minus': False, 'theta_plus': False, 'theta_minus': False,
                                           'theta': True, 'delta': True, 'eta': True, 'gamma': False,
                                           'gamma_plus0': True, 'gamma_minus0': True, 'theta_plus0': True, 'theta_minus0': True}

    def mean_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        I_plus = (-I + self._theta_plus) / np.sqrt(self._gamma_plus)
        I_minus = (I + self._theta_minus) / np.sqrt(self._gamma_minus)

        etg_plus = erf_times_gauss(I_plus)
        etg_minus = erf_times_gauss(I_minus)

        p_plus = 1 / (1 + (etg_minus / np.sqrt(self._gamma_minus)
                           ) / (etg_plus / np.sqrt(self._gamma_plus)))
        nans = np.isnan(p_plus)
        p_plus[nans] = 1.0 * (np.abs(I_plus[nans]) > np.abs(I_minus[nans]))
        p_minus = 1 - p_plus
        mean_pos = (-I_plus + 1 / etg_plus) / np.sqrt(self._gamma_plus)
        mean_neg = (I_minus - 1 / etg_minus) / np.sqrt(self._gamma_minus)
        return mean_pos * p_plus + mean_neg * p_minus

    def mean2_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        I_plus = (-I + self._theta_plus) / np.sqrt(self._gamma_plus)
        I_minus = (I + self._theta_minus) / np.sqrt(self._gamma_minus)

        etg_plus = erf_times_gauss(I_plus)
        etg_minus = erf_times_gauss(I_minus)

        p_plus = 1 / (1 + (etg_minus / np.sqrt(self._gamma_minus)
                           ) / (etg_plus / np.sqrt(self._gamma_plus)))
        nans = np.isnan(p_plus)
        p_plus[nans] = 1.0 * (np.abs(I_plus[nans]) > np.abs(I_minus[nans]))
        p_minus = 1 - p_plus
        mean2_pos = 1 / self._gamma_plus * (1 + I_plus**2 - I_plus / etg_plus)
        mean2_neg = 1 / self._gamma_minus * \
            (1 + I_minus**2 - I_minus / etg_minus)
        return mean2_pos * p_plus + mean2_neg * p_minus

    def mean12_pm_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        I_plus = (-I + self._theta_plus) / np.sqrt(self._gamma_plus)
        I_minus = (I + self._theta_minus) / np.sqrt(self._gamma_minus)

        etg_plus = erf_times_gauss(I_plus)
        etg_minus = erf_times_gauss(I_minus)

        p_plus = 1 / (1 + (etg_minus / np.sqrt(self._gamma_minus)
                           ) / (etg_plus / np.sqrt(self._gamma_plus)))
        nans = np.isnan(p_plus)
        p_plus[nans] = 1.0 * (np.abs(I_plus[nans]) > np.abs(I_minus[nans]))
        p_minus = 1 - p_plus
        mean_pos = (-I_plus + 1 / etg_plus) / np.sqrt(self._gamma_plus)
        mean_neg = (I_minus - 1 / etg_minus) / np.sqrt(self._gamma_minus)
        mean2_pos = 1 / self._gamma_plus * (1 + I_plus**2 - I_plus / etg_plus)
        mean2_neg = 1 / self._gamma_minus * \
            (1 + I_minus**2 - I_minus / etg_minus)
        return (p_plus * mean_pos, p_minus * mean_neg, p_plus * mean2_pos, p_minus * mean2_neg)

    def var_from_inputs(self, I, I0=None, beta=1):
        (mu_pos, mu_neg, mu2_pos, mu2_neg) = self.mean12_pm_from_inputs(
            I, I0=I0, beta=beta)
        return (mu2_pos + mu2_neg) - (mu_pos + mu_neg)**2

    def cgf_from_inputs(self, I, I0=None, beta=1):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        if type(beta) == np.ndarray:
            return cy_utilities.cgf_from_inputs_dReLU_numba3(I, self._gamma_plus[:, 0, :],
                                                             self._gamma_minus[:, 0, :],
                                                             self._theta_plus[:, 0, :],
                                                             self._theta_minus[:, 0, :])
        else:
            return cy_utilities.cgf_from_inputs_dReLU_numba2(I, self._gamma_plus[0],
                                                             self._gamma_minus[0],
                                                             self._theta_plus[0],
                                                             self._theta_minus[0])

    def transform(self, I):
        self.get_params(beta=1)
        return ((I + self._theta_minus) * (I <= np.minimum(-self._theta_minus, (self._theta_plus / np.sqrt(self._gamma_plus) - self._theta_minus / np.sqrt(self._gamma_minus)) / (1 / np.sqrt(self._gamma_plus) + 1 / np.sqrt(self._gamma_minus))))) / self._gamma_minus \
            + ((I - self._theta_plus) * (I >= np.maximum(self._theta_plus, (self._theta_plus / np.sqrt(self._gamma_plus) - self._theta_minus /
                                                                            np.sqrt(self._gamma_minus)) / (1 / np.sqrt(self._gamma_plus) + 1 / np.sqrt(self._gamma_minus))))) / self._gamma_plus

    def sample_from_inputs(self, I, I0=None, beta=1, out=None, **kwargs):
        I = self.get_input(I, I0=I0, beta=beta)
        self.get_params(beta=beta)
        if out is None:
            out = np.empty_like(I)
        if type(beta) == np.ndarray:
            cy_utilities.sample_from_inputs_dReLU_numba3(I, self._gamma_plus[:, 0, :],
                                                         self._gamma_minus[:, 0, :],
                                                         self._theta_plus[:, 0, :],
                                                         self._theta_minus[:, 0, :], out)
        else:
            cy_utilities.sample_from_inputs_dReLU_numba2(I, self._gamma_plus[0],
                                                         self._gamma_minus[0],
                                                         self._theta_plus[0],
                                                         self._theta_minus[0], out)
        return out

    def energy(self, config, remove_init=False, beta=1):
        config_plus = np.maximum(config, 0)
        config_minus = np.maximum(-config, 0)
        if remove_init:
            return np.dot(config_plus**2, self.gamma_plus - self.gamma_plus0) / 2. + np.dot(config_minus**2, self.gamma_minus - self.gamma_minus0) / 2. + np.dot(config_plus, self.theta_plus - self.theta_plus0) + np.dot(config_minus, self.theta_minus - self.theta_minus0)
        else:
            self.get_params(beta=beta)
            return (config_plus**2 * self._gamma_plus).sum(-1) / 2. + (config_minus**2 * self._gamma_minus).sum(-1) / 2. + (config_plus * self._theta_plus).sum(-1) + (config_minus * self._theta_minus).sum(-1)

    # dReLU.
    def get_moments(self, data, I0=None, value='input', weights=None, beta=1):
        if value == 'input':
            mu_pos, mu_neg, mu2_pos, mu2_neg = self.mean12_pm_from_inputs(
                data, I0=I0, beta=beta)
            mu_pos = average(mu_pos, weights=weights)
            mu_neg = average(mu_neg, weights=weights)
            mu2_pos = average(mu2_pos, weights=weights)
            mu2_neg = average(mu2_neg, weights=weights)
        else:
            mu_pos = average(np.maximum(data, 0), weights=weights)
            mu_neg = average(np.minimum(data, 0), weights=weights)
            mu2_pos = average(np.maximum(data, 0)**2, weights=weights)
            mu2_neg = average(np.minimum(data, 0)**2, weights=weights)
        return (mu2_pos, mu2_neg, mu_pos, mu_neg)

    def internal_gradients(self, *args, **kwargs):  # dReLU
        gradients = super(dReLULayer, self).internal_gradients(*args, **kwargs)
        if self.position == 'hidden':
            gradients['gamma'] = gradients['gamma_plus'] / \
                (1 + self.eta) + gradients['gamma_minus'] / (1 - self.eta)
            gradients['theta'] = gradients['theta_plus'] - \
                gradients['theta_minus']
            gradients['delta'] = gradients['theta_plus'] / \
                np.sqrt(1 + self.eta) + \
                gradients['theta_minus'] / np.sqrt(1 - self.eta)
            gradients['eta'] = (- self.gamma / (1 + self.eta)**2 * gradients['gamma_plus']
                                + self.gamma / (1 - self.eta)**2 *
                                gradients['gamma_minus']
                                - self.theta /
                                (2 * np.sqrt(1 + self.eta)**3) *
                                gradients['theta_plus']
                                + self.theta / (2 * np.sqrt(1 - self.eta)**3) * gradients['theta_minus'])
        return gradients

    def init_params_from_data(self, X, eps=1e-6, value='data', weights=None):
        if X is None:
            self.gamma_plus = np.ones(self.N, dtype=curr_float)
            self.gamma_plus0 = np.ones(self.N, dtype=curr_float)
            self.gamma_minus = np.ones(self.N, dtype=curr_float)
            self.gamma_minus0 = np.ones(self.N, dtype=curr_float)
            self.theta_plus = np.zeros(self.N, dtype=curr_float)
            self.theta_plus0 = np.zeros(self.N, dtype=curr_float)
            self.theta_minus = np.zeros(self.N, dtype=curr_float)
            self.theta_minus0 = np.zeros(self.N, dtype=curr_float)
            # batch norm parametrization.
            self.gamma = np.ones(self.N, dtype=curr_float)
            # batch norm parametrization.
            self.theta = np.zeros(self.N, dtype=curr_float)
            # batch norm parametrization.
            self.delta = np.zeros(self.N, dtype=curr_float)
            # batch norm parametrization.
            self.eta = np.zeros(self.N, dtype=curr_float)
        else:
            if value == 'moments':
                moments = X
            else:
                moments = self.get_moments(X, value=value, weights=weights)
            mean = moments[2] + moments[3]
            var = moments[0] + moments[1] - (moments[2] + moments[3])**2
            self.gamma_plus = 1 / (var + eps)
            self.gamma_minus = self.gamma_plus.copy()
            self.theta_plus = - self.gamma_plus * mean
            self.theta_minus = - self.theta_plus
            self.gamma_plus0 = self.gamma_plus.copy()
            self.gamma_minus0 = self.gamma_minus.copy()
            self.theta_plus0 = self.theta_plus.copy()
            self.theta_minus0 = self.theta_minus.copy()

    def batch_norm_update(self, mu_I, I, lr=1, weights=None):
        delta_mu_I = (mu_I - self.mu_I)
        self.mu_I = mu_I
        self.theta += delta_mu_I
        self.theta_plus += delta_mu_I
        self.theta_minus -= delta_mu_I
        e = self.mean_from_inputs(I) * self.gamma[np.newaxis, :]
        v = (self.var_from_inputs(I) * self.gamma[np.newaxis, :] - 1)
        var_e = average(e**2, weights=weights) - average(e, weights=weights)**2
        mean_v = average(v, weights=weights)
        new_gamma = (1 + mean_v + np.sqrt((1 + mean_v)**2 + 4 * var_e)) / 2
        gamma_min = np.maximum(np.maximum(
            self.gamma_min,  # gamma cannot be too small
            self.gamma_drop_max * self.gamma),  # cannot drop too quickly.
            # The jump cannot be too large.
            np.maximum(-self.delta, 0) * (np.sqrt(1 - self.eta) + \
                                          np.sqrt(1 + self.eta)) / self.jump_max
        )
        self.gamma = np.maximum(
            (1 - lr) * self.gamma + lr * new_gamma, gamma_min)

        self.gamma_plus = self.gamma / (1 + self.eta)
        self.gamma_minus = self.gamma / (1 - self.eta)

    def batch_norm_update_gradient(self, gradient_W, gradient_hlayer, V, I, mu, n_c, weights=None):
        B = V.shape[0]
        if weights is None:
            weights = np.ones(B, dtype=curr_float)
        if n_c == 1:
            V = np.asarray(V, dtype=curr_float)
            dtheta_dw, dgamma_dtheta, dgamma_ddelta, dgamma_deta, dgamma_dw = cy_utilities.get_cross_derivatives_dReLU_numba(
                V, I, self.gamma, self.theta, self.eta, self.delta, weights)
        else:
            dtheta_dw, dgamma_dtheta, dgamma_ddelta, dgamma_deta, dgamma_dw = cy_utilities.get_cross_derivatives_dReLU_Potts_numba(
                V, I, self.gamma, self.theta, self.eta, self.delta, weights, n_c)

        add_to_gradient(gradient_hlayer['theta'],
                        gradient_hlayer['gamma'], dgamma_dtheta)
        add_to_gradient(gradient_hlayer['delta'],
                        gradient_hlayer['gamma'], dgamma_ddelta)
        add_to_gradient(gradient_hlayer['eta'],
                        gradient_hlayer['gamma'], dgamma_deta)
        add_to_gradient(gradient_W, gradient_hlayer['theta'], dtheta_dw)
        add_to_gradient(gradient_W, gradient_hlayer['gamma'], dgamma_dw)
        return

    def recompute_params(self, which='regular'):
        if which == 'regular':
            saturate(self.eta, 0.95)
            self.gamma_plus = self.gamma / (1 + self.eta)
            self.gamma_minus = self.gamma / (1 - self.eta)
            self.theta_plus = self.theta + self.delta / np.sqrt(1 + self.eta)
            self.theta_minus = -self.theta + self.delta / np.sqrt(1 - self.eta)
        else:
            self.gamma = 2. / (1. / self.gamma_plus + 1. / self.gamma_minus)
            self.eta = (self.gamma / self.gamma_plus -
                        self.gamma / self.gamma_minus) / 2
            self.delta = (self.theta_plus + self.theta_minus) / \
                (1 / np.sqrt(1 + self.eta) + 1 / np.sqrt(1 - self.eta))
            self.theta = self.theta_plus - self.delta / np.sqrt(1 + self.eta)


class Bernoulli_coupledLayer(Layer):
    def __init__(self, N=100, position='visible', zero_field=False, batch_norm=False, random_state=None, **kwargs):
        super(Bernoulli_coupledLayer, self).__init__(N=N, nature='Bernoulli_coupled',
                                                     position=position, batch_norm=batch_norm, n_c=1, random_state=random_state)
        self.fields = np.zeros(self.N, dtype=curr_float)
        self.fields0 = np.zeros(self.N, dtype=curr_float)  # useful for PT.
        self.couplings = np.zeros([self.N, self.N], dtype=curr_float)
        self.couplings0 = np.zeros([self.N, self.N], dtype=curr_float)
        self.zero_field = zero_field
        self.list_params = ['fields', 'couplings']
        self.factors = [1., 1.]
        self.params_anneal = {'fields': True, 'couplings': True}
        self.params_newaxis = {'fields': False, 'couplings': False}

        self.do_grad_updates = {'fields':  ~self.zero_field, 'couplings': True,
                                'fields0': ~self.zero_field, 'couplings0': False}
        self.do_grad_updates_batch_norm = self.do_grad_updates

    def sample_from_inputs(self, I, I0=None, beta=1, previous=(None, None), **kwargs):
        if I is None:
            if I0 is not None:
                I = (1 - beta) * I0
        else:
            I = self.get_input(I, I0=I0, beta=beta)
        (x, fields_eff) = previous
        if x is None:
            B = I.shape[0]
            x = self.random_state.randint(
                0, high=2, size=[B, self.N]).astype(self.type)
        else:
            B = x.shape[0]
        if fields_eff is None:
            fields_eff = self.fields[np.newaxis] + \
                self.compute_output(x, self.couplings)

        if I is not None:
            x, fields_eff = cy_utilities.Bernoulli_Gibbs_input_C(
                x, fields_eff, I, B, self.N, self.fields0, self.couplings, beta)
        else:
            x, fields_eff = cy_utilities.Bernoulli_Gibbs_free_C(
                x, fields_eff, B, self.N, self.fields0, self.couplings, beta)
        return (x, fields_eff)

    def energy(self, config, beta=1, remove_init=False):
        if remove_init:
            fields = self.fields - self.fields0
            couplings = self.couplings
        else:
            if beta == 1:
                fields = self.fields
                couplings = self.couplings
            else:
                fields = beta * self.fields + (1 - beta) * self.fields0
                couplings = beta * self.couplings

        return - np.dot(config, fields) - 0.5 * (np.dot(config, couplings) * config).sum(1)

    def init_params_from_data(self, X, eps=1e-6, value='data', weights=None):
        if X is None:
            self.fields = np.zeros(self.N, dtype=curr_float)
            self.fields0 = np.zeros(self.N, dtype=curr_float)  # useful for PT.
            self.couplings = np.zeros([self.N, self.N], dtype=curr_float)
            self.couplings0 = np.zeros([self.N, self.N], dtype=curr_float)
        else:
            if value == 'moments':
                moments = X
            else:
                moments = self.get_moments(X, value=value, weights=weights)

            self.fields = np.log((moments[0] + eps) / (1 - moments[0] + eps))
            self.fields0 = self.fields.copy()
            self.couplings *= 0
            self.couplings0 *= 0

    # BernoulliCoupled.
    def get_moments(self, data, I0=None, value='input', weights=None, beta=1):
        if value == 'input':
            print('get moments from input not supported BernoulliCoupled')
            return
        elif value == 'data':
            mu = average(data, weights=weights)
            comu = average_product(data, data, weights=weights)
        return (mu, comu)

    def internal_gradients(self, *args, l2=0, l1=0, **kwargs):  # Bernoulli_coupled:
        gradients = super(Bernoulli_coupledLayer,
                          self).internal_gradients(*args,**kwargs)
        if l2 > 0:
            gradients['couplings'] -= l2 * self.couplings
        if l1 > 0:
            gradients['couplings'] -= l1 * np.sign(self.couplings)
        return gradients


class Spin_coupledLayer(Layer):
    def __init__(self, N=100, position='visible', batch_norm=False, zero_field=False, random_state=None, **kwargs):
        super(Spin_coupledLayer, self).__init__(N=N, nature='Spin_coupled',
                                                position=position, batch_norm=batch_norm, n_c=1, random_state=random_state)
        self.fields = np.zeros(self.N, dtype=curr_float)
        self.fields0 = np.zeros(self.N, dtype=curr_float)  # useful for PT.
        self.couplings = np.zeros([self.N, self.N], dtype=curr_float)
        self.couplings0 = np.zeros([self.N, self.N], dtype=curr_float)

        self.zero_field = zero_field
        self.list_params = ['fields', 'couplings']
        self.factors = [1., 1.]
        self.params_anneal = {'fields': True, 'couplings': True}
        self.params_newaxis = {'fields': False, 'couplings': False}

        self.do_grad_updates = {'fields': ~zero_field, 'couplings': True,
                                'fields0': ~zero_field, 'couplings0': False}
        self.do_grad_updates_batch_norm = self.do_grad_updates

    def sample_from_inputs(self, I, I0=None, beta=1, previous=(None, None), **kwargs):
        if I is None:
            if I0 is not None:
                I = (1 - beta) * I0
        else:
            I = self.get_input(I, I0=I0, beta=beta)
        (x, fields_eff) = previous

        if x is None:
            B = I.shape[0]
            x = (2 * self.random_state.randint(0, high=2,
                                               size=[B, self.N]) - 1).astype(self.type)
        else:
            B = x.shape[0]

        if fields_eff is None:
            fields_eff = self.fields[np.newaxis] + \
                self.compute_output(x, self.couplings)

        if I is not None:
            x, fields_eff = cy_utilities.Spin_Gibbs_input_C(
                x, fields_eff, I, B, self.N, self.fields0, self.couplings, beta)
        else:
            x, fields_eff = cy_utilities.Spin_Gibbs_free_C(
                x, fields_eff, B, self.N, self.fields0, self.couplings, beta)
        return (x, fields_eff)

    def energy(self, config, beta=1, remove_init=False):
        if remove_init:
            fields = self.fields - self.fields0
            couplings = self.couplings
        else:
            if beta == 1:
                fields = self.fields
                couplings = self.couplings
            else:
                fields = beta * self.fields + (1 - beta) * self.fields0
                couplings = beta * self.couplings
        return - np.dot(config, fields) - 0.5 * (np.dot(config, couplings) * config).sum(1)

    def init_params_from_data(self, X, eps=1e-6, value='data', weights=None):
        if X is None:
            self.fields = np.zeros(self.N, dtype=curr_float)
            self.fields0 = np.zeros(self.N, dtype=curr_float)  # useful for PT.
            self.couplings = np.zeros([self.N, self.N], dtype=curr_float)
            self.couplings0 = np.zeros([self.N, self.N], dtype=curr_float)
        else:
            if value == 'moments':
                moments = X
            else:
                moments = self.get_moments(X, value=value, weights=weights)
            self.fields = 0.5 * \
                np.log((1 + moments[0] + eps) / (1 - moments[0] + eps))
            self.fields0 = self.fields.copy()
            self.couplings *= 0
            self.couplings0 *= 0

    # BernoulliCoupled.
    def get_moments(self, data, I0=None, value='input', weights=None, beta=1):
        if value == 'input':
            print('get moments from input not supported BernoulliCoupled')
            return
        elif value == 'data':
            mu = average(data, weights=weights)
            comu = average_product(data, data, weights=weights)
        return (mu, comu)

    def internal_gradients(self, *args, l2=0, l1=0, **kwargs):  # Spin_coupled:
        gradients = super(Spin_coupledLayer, self).internal_gradients(*args,**kwargs)
        if l2 > 0:
            gradients['couplings'] -= l2 * self.couplings
        if l1 > 0:
            gradients['couplings'] -= l1 * np.sign(self.couplings)
        return gradients


class Potts_coupledLayer(Layer):
    def __init__(self, N=100, position='visible', batch_norm=False, zero_field=False, gauge='zerosum', n_c=2, random_state=None, **kwargs):
        super(Potts_coupledLayer, self).__init__(N=N, nature='Potts_coupled',
                                                 position=position, batch_norm=batch_norm, n_c=n_c, random_state=random_state)
        self.fields = np.zeros([self.N, self.n_c], dtype=curr_float)
        self.fields0 = np.zeros([self.N, self.n_c], dtype=curr_float)
        self.couplings = np.zeros(
            [self.N, self.N, self.n_c, self.n_c], dtype=curr_float)
        self.couplings0 = np.zeros(
            [self.N, self.N, self.n_c, self.n_c], dtype=curr_float)
        self.gauge = gauge
        self.zero_field = zero_field

        self.list_params = ['fields', 'couplings']
        self.factors = [1., 1.]
        self.params_anneal = {'fields': True, 'couplings': True}
        self.params_newaxis = {'fields': False, 'couplings': False}

        self.do_grad_updates = {'fields': self.zero_field, 'couplings': True,
                                'fields0': self.zero_field, 'couplings0': False}
        self.do_grad_updates_batch_norm = self.do_grad_updates

    def sample_from_inputs(self, I, I0=None, beta=1, previous=(None, None), **kwargs):
        if I is None:
            if I0 is not None:
                I = (1 - beta) * I0
        else:
            I = self.get_input(I, I0=I0, beta=beta)
        (x, fields_eff) = previous
        if x is None:
            B = I.shape[0]
            x = self.random_state.randint(0, high=self.n_c, size=[
                                          B, self.N]).astype(self.type)
        else:
            B = x.shape[0]
        if fields_eff is None:
            fields_eff = self.fields[np.newaxis] + \
                self.compute_output(x, self.couplings)

        if I is not None:
            x, fields_eff = cy_utilities.Potts_Gibbs_input_C(
                x, fields_eff, I, B, self.N, self.n_c, self.fields0, self.couplings, beta)
        else:
            x, fields_eff = cy_utilities.Potts_Gibbs_free_C(
                x, fields_eff, B, self.N, self.n_c, self.fields0, self.couplings, beta)
        return (x, fields_eff)

    def energy(self, config, beta=1, remove_init=False):
        if remove_init:
            fields = self.fields - self.fields0
            couplings = self.couplings
        else:
            if beta == 1:
                fields = self.fields
                couplings = self.couplings
            else:
                fields = beta * self.fields + (1 - beta) * self.fields0
                couplings = beta * self.couplings
        return - cy_utilities.dot_Potts_C(config, fields) - 0.5 * bilinear_form(couplings, config, config, c1=self.n_c, c2=self.n_c)

    def init_params_from_data(self, X, eps=1e-6, value='data', weights=None):
        if X is None:
            self.fields = np.zeros([self.N, self.n_c], dtype=curr_float)
            self.fields0 = np.zeros([self.N, self.n_c], dtype=curr_float)
            self.couplings = np.zeros(
                [self.N, self.N, self.n_c, self.n_c], dtype=curr_float)
            self.couplings0 = np.zeros(
                [self.N, self.N, self.n_c, self.n_c], dtype=curr_float)
        else:
            if value == 'moments':
                moments = X
            else:
                moments = self.get_moments(X, value=value, weights=weights)
            self.fields = invert_softmax(moments[0], eps=eps, gauge=self.gauge)
            self.fields0 = self.fields.copy()
            self.couplings *= 0
            self.couplings0 *= 0

    # PottsCoupled.
    def get_moments(self, data, I0=None, value='input', weights=None, beta=1):
        if value == 'input':
            print('get moments from input not supported SpinCoupled')
            return
        elif value == 'data':
            mu = average(data, weights=weights, c=self.n_c)
            comu = average_product(data, data, weights=weights, c=self.n_c)
        return (mu, comu)

    def internal_gradients(self, *args,l2=0, l1=0, **kwargs):  # Potts_coupled:
        gradients = super(Potts_coupledLayer,
                          self).internal_gradients(*args,**kwargs)
        if l2 > 0:
            gradients['couplings'] -= l2 * self.couplings
        if l1 > 0:
            gradients['couplings'] -= l1 * np.sign(self.couplings)
        return gradients


class InterpolateLayer(Layer):
    def __init__(self, N=100, nature='Bernoulli', degree=2, position='visible', batch_norm=False, n_c=1, random_state=None, **kwargs):
        super(InterpolateLayer, self).__init__(N=N, nature=nature, position=position,
                                               batch_norm=batch_norm, n_c=n_c, random_state=random_state)
        self.degree = degree
        for key in self.list_params:
            self.__dict__[
                key + '1'] = np.zeros([degree - 1] + list(self.__dict__[key].shape), dtype=curr_float)
            self.do_grad_updates[key + '1'] = self.do_grad_updates[key + '0']
            self.do_grad_updates_batch_norm[key +
                                            '1'] = self.do_grad_updates[key + '0']

    def get_coefficients(self, beta=1):
        if not hasattr(self, 'Ck'):
            self.Ck = np.array([4 * (k + 2.0) / 2.0 * ((k + 2.0) / (k + 1e-10))**(k / 2.0)
                                for k in range(self.degree - 1)], dtype=curr_float)
        return np.array([self.Ck[k] * beta * (1.0 - beta) * ((beta - 0.5) * 2)**k for k in range(self.degree - 1)], dtype=curr_float)

    def get_params(self, beta=1):
        beta_is_array = (type(beta) == np.ndarray)
        if not beta_is_array:
            beta_is_one = (beta == 1)
            beta_is_zero = (beta == 0)
        else:
            beta_is_zero = False
            beta_is_one = False

        if not (beta_is_one | beta_is_zero):
            coefficients = self.get_coefficients(beta)

        if beta_is_array:
            if self.n_c > 1:
                beta = beta[:, np.newaxis, np.newaxis]
            else:
                beta = beta[:, np.newaxis]

        for key in self.list_params:
            if beta_is_one:
                tmp = getattr(self, key)
            elif beta_is_zero:
                tmp = getattr(self, key + '0')
            else:
                if beta_is_array:
                    tmp = beta * getattr(self, key)[np.newaxis] + (1 - beta) * getattr(self, key + '0')[
                        np.newaxis] + np.tensordot(coefficients, getattr(self, key + '1'), (0, 0))
                else:
                    tmp = beta * getattr(self, key) + (1 - beta) * getattr(
                        self, key + '0') + np.tensordot(coefficients, getattr(self, key + '1'), (0, 0))

            if self.params_newaxis[key]:
                if beta_is_array:
                    tmp = tmp[:, np.newaxis]
                else:
                    tmp = tmp[np.newaxis, :]
            setattr(self, '_' + key, tmp)

    def internal_gradients_interpolation(self, datas, betas, value='data', gradient=None):
        if gradient is None:
            gradient = {}
        if not hasattr(self, '_target_moments0'):
            self._target_moments0 = self.get_moments(None, beta=0)
            self._mean_weight0 = 1.

        if (self.degree < 2) | len(betas) < 3:
            pass
        else:
            nbetas = len(betas)
            moments = [self.get_moments(datas[l], value=value)
                       for l in range(1, nbetas - 1)]
            coefficients = self.get_coefficients(betas)[:, 1:-1]
            coefficients_sum = coefficients.sum(-1)
            if self.n_c > 1:
                coefficients_sum = coefficients_sum[:, np.newaxis, np.newaxis]
            else:
                coefficients_sum = coefficients_sum[:, np.newaxis]
            for k, key in enumerate(self.list_params):
                gradient[key + '1'] = self._mean_weight0 * self.factors[k] * (coefficients_sum * self._target_moments0[k][np.newaxis] - np.tensordot(
                    coefficients,  np.array([moment[k] for moment in moments]), axes=(1, 0)))
        return gradient


class BernoulliInterpolateLayer(InterpolateLayer, BernoulliLayer):
    pass


class SpinInterpolateLayer(InterpolateLayer, SpinLayer):
    pass


class PottsInterpolateLayer(InterpolateLayer, PottsLayer):
    pass


class GaussianInterpolateLayer(InterpolateLayer, GaussianLayer):
    pass


class ReLUplusInterpolateLayer(InterpolateLayer, ReLUplusLayer):
    pass


class ReLUInterpolateLayer(InterpolateLayer, ReLULayer):
    pass


class dReLUInterpolateLayer(InterpolateLayer, dReLULayer):
    pass
