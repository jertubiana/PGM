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
import layer
import pgm
import moi
import utilities
from utilities import check_random_state, gen_even_slices, logsumexp, log_logistic, average, average_product, saturate, check_nan, get_permutation
import time
import copy

from float_precision import double_precision, curr_float, curr_int


# %%


class RBM(pgm.PGM):
    def __init__(self, n_v=100, n_h=20, visible='Bernoulli', hidden='Bernoulli', interpolate=False, degree_interpolate=5, n_cv=1, n_ch=1, random_state=None, gauge='zerosum', zero_field=False):
        self.n_v = n_v
        self.n_h = n_h
        self.n_visibles = n_v
        self.n_hiddens = n_h
        self.visible = visible
        self.hidden = hidden
        self.random_state = check_random_state(random_state)
        if self.visible == 'Potts':
            self.n_cv = n_cv
        else:
            self.n_cv = 1
        if self.hidden == 'Potts':
            self.n_ch = n_ch
        else:
            self.n_ch = 1

        super(RBM, self).__init__(n_layers=2, layers_size=[self.n_v, self.n_h], layers_nature=[
            visible, hidden], layers_n_c=[self.n_cv, self.n_ch], layers_name=['vlayer', 'hlayer'])

        self.gauge = gauge
        self.zero_field = zero_field
        self.interpolate = interpolate
        self.degree_interpolate = degree_interpolate
        if self.interpolate:
            self.vlayer = layer.initLayer(N=self.n_v, nature=self.visible + 'Interpolate', degree=degree_interpolate,
                                          position='visible', n_c=self.n_cv, random_state=self.random_state, zero_field=self.zero_field)
            self.hlayer = layer.initLayer(N=self.n_h, nature=self.hidden + 'Interpolate', degree=degree_interpolate,
                                          position='hidden', n_c=self.n_ch, random_state=self.random_state, zero_field=self.zero_field)
        else:
            self.vlayer = layer.initLayer(N=self.n_v, nature=self.visible, position='visible',
                                          n_c=self.n_cv, random_state=self.random_state, zero_field=self.zero_field)
            self.hlayer = layer.initLayer(N=self.n_h, nature=self.hidden, position='hidden',
                                          n_c=self.n_ch, random_state=self.random_state, zero_field=self.zero_field)
        self.init_weights(0.01)
        self.tmp_l2_fields = 0
        self._Ivh = None
        self._Ihv = None
        self._Ivz = None
        self._Izv = None
        self._Ihz = None
        self._Izh = None

    def init_weights(self, amplitude):
        if (self.n_ch > 1) & (self.n_cv > 1):
            self.weights = amplitude * \
                self.random_state.randn(
                    self.n_h, self.n_v, self.n_ch, self.n_cv)
            self.weights = pgm.gauge_adjust_couplings(
                self.weights, self.n_ch, self.n_cv, gauge=self.gauge)
        elif (self.n_ch > 1) & (self.n_cv == 1):
            self.weights = amplitude * \
                self.random_state.randn(self.n_h, self.n_v, self.n_ch)
            self.weights = pgm.gauge_adjust_couplings(
                self.weights, self.n_ch, self.n_cv, gauge=self.gauge)
        elif (self.n_ch == 1) & (self.n_cv > 1):
            self.weights = amplitude * \
                self.random_state.randn(self.n_h, self.n_v, self.n_cv)
            self.weights = pgm.gauge_adjust_couplings(
                self.weights, self.n_ch, self.n_cv, gauge=self.gauge)
        else:
            self.weights = amplitude * \
                self.random_state.randn(self.n_h, self.n_v)
        self.weights = np.asarray(self.weights, dtype=curr_float)

    def markov_step(self, x, beta=1, recompute=True):
        (v, h) = x
        if recompute:
            self._Ivh = self.vlayer.compute_output(
                v, self.weights, direction='up', out=self._Ivh)
        h = self.hlayer.sample_from_inputs(self._Ivh, beta=beta, out=h)
        self._Ihv = self.hlayer.compute_output(
            h, self.weights, direction='down', out=self._Ihv)
        v = self.vlayer.sample_from_inputs(self._Ihv, beta=beta, out=v)
        return (v, h)

    def markov_step_h(self, x, beta=1, recompute=True):
        (v, h) = x
        if recompute:
            self._Ihv = self.hlayer.compute_output(
                h, self.weights, direction='down', out=self._Ihv)
        v = self.vlayer.sample_from_inputs(self._Ihv, beta=beta, out=v)
        self._Ivh = self.vlayer.compute_output(
            v, self.weights, direction='up', out=self._Ivh)
        h = self.hlayer.sample_from_inputs(self._Ivh, beta=beta, out=h)
        return (v, h)

    def markov_step_and_energy(self, x, E, beta=1):
        (v, h) = x
        self._Ivh = self.vlayer.compute_output(
            v, self.weights, direction='up', out=self._Ivh)
        h = self.hlayer.sample_from_inputs(self._Ivh, beta=beta, out=h)
        self._Ihv = self.hlayer.compute_output(
            h, self.weights, direction='down', out=self._Ihv)
        v, E = self.vlayer.sample_and_energy_from_inputs(
            self._Ihv, beta=beta, remove_init=True)
        E += self.hlayer.energy(h, remove_init=True)
        return (v, h), E

    def markov_step_APT(self, x,  beta=1, recompute=True):
        (v, h, z) = x
        beta_is_array = (type(beta) == np.ndarray)
        if beta_is_array:
            N_PT = v.shape[0]
            B = v.shape[1]
        else:
            N_PT = 1
            B = v.shape[0]
        if recompute:
            self._Ivz = self.vlayer.compute_output(
                v, self.weights_MoI, direction='up', out=self._Ivz)
            self._Ivh = self.vlayer.compute_output(
                v, self.weights, direction='up', out=self._Ivh)
        h = self.hlayer.sample_from_inputs(self._Ivh, beta=beta)
        # many temperatures. Last one at beta=0 = MoI configuration, obtained by direct sampling.
        if beta_is_array:
            z[-1] = self.zlayer.sample_from_inputs(
                np.zeros([B, 1, self.zlayer.n_c], dtype=curr_float), beta=1)
            z[:-1] = self.zlayer.sample_from_inputs(np.zeros(
                [N_PT - 1, B, 1, self.zlayer.n_c], dtype=curr_float), I0=self._Ivz[:-1], beta=beta[:-1])
        else:
            if beta == 0:
                z = self.zlayer.sample_from_inputs(
                    np.zeros([B, 1, self.zlayer.n_c], dtype=curr_float), beta=1)
            else:
                z = self.zlayer.sample_from_inputs(
                    np.zeros([B, 1, self.zlayer.n_c], dtype=curr_float), I0=self._Ivz, beta=beta)

        self._Izv = self.zlayer.compute_output(
            z, self.weights_MoI, direction='down', out=self._Izv)
        self._Ihv = self.hlayer.compute_output(
            h, self.weights, direction='down', out=self._Ihv)
        v = self.vlayer.sample_from_inputs(self._Ihv, I0=self._Izv, beta=beta)
        return (v, h, z)

    def markov_step_APTh(self, x,  beta=1, recompute=True):
        (v, h, z) = x
        beta_is_array = (type(beta) == np.ndarray)
        if beta_is_array:
            N_PT = h.shape[0]
            B = h.shape[1]
        else:
            N_PT = 1
            B = h.shape[0]
        if recompute:
            self._Ihz = self.hlayer.compute_output(
                h, self.weights_MoI, direction='up', out=self._Ihz)
            self._Ihv = self.hlayer.compute_output(
                h, self.weights, direction='down', out=self._Ihz)
        v = self.vlayer.sample_from_inputs(self._Ihv, beta=beta)
        if beta_is_array:
            z[-1] = self.zlayer.sample_from_inputs(
                np.zeros([B, 1, self.zlayer.n_c], dtype=curr_float), beta=1)
            z[:-1] = self.zlayer.sample_from_inputs(np.zeros(
                [N_PT - 1, B, 1, self.zlayer.n_c], dtype=curr_float), I0=self._Ihz[:-1], beta=beta[:-1])
        else:
            if beta == 0:
                z = self.zlayer.sample_from_inputs(
                    np.zeros([B, 1, self.zlayer.n_c], dtype=curr_float), beta=1)
            else:
                z = self.zlayer.sample_from_inputs(
                    np.zeros([B, 1, self.zlayer.n_c], I0=self._Ihz, dtype=curr_float), beta=beta)

        self._Ivh = self.vlayer.compute_output(
            v, self.weights, direction='up', out=self._Ivh)
        self._Izh = self.zlayer.compute_output(
            z, self.weights_MoI, direction='down', out=self._Izh)
        h = self.hlayer.sample_from_inputs(self._Ivh, I0=self._Izh, beta=beta)
        return (v, h, z)

    def exchange_step_PT(self, x, E, record_acceptance=True, compute_energy=True):
        (v, h) = x
        if compute_energy:
            E = self.energy((v, h), remove_init=True)
        # 0<->1, 2<->3,... if count even, 0<->0, 1<->2,3<->4,... if count odd.
        permutation = get_permutation(self.N_PT, self.count_swaps)
        F = self.betas[:, np.newaxis] * E
        F_swapped = self.betas[permutation, np.newaxis] * E
        (v, h, E) = self.exchange_step((v, h, E), permutation, F, F_swapped)
        return (v, h), E

    def exchange_step_PTv(self, x):
        (v, h) = x
        self._Ivh = self.vlayer.compute_output(
            v, self.weights, direction='up', out=self._Ivh)
        # 0<->1, 2<->3,... if count even, 0<->0, 1<->2,3<->4,... if count odd.
        permutation = get_permutation(self.N_PT, self.count_swaps)
        F = self.free_energy(v, Ivh=self._Ivh, beta=self.betas)
        F_swapped = self.free_energy(
            v[permutation], Ivh=self._Ivh[permutation], beta=self.betas)
        (v, h, self._Ivh) = self.exchange_step(
            (v, h, self._Ivh), permutation, F, F_swapped)
        return (v, h)

    def exchange_step_PTh(self, x):
        (v, h) = x
        self._Ihv = self.hlayer.compute_output(
            h, self.weights, direction='down', out=self._Ihv)
        # 0<->1, 2<->3,... if count even, 0<->0, 1<->2,3<->4,... if count odd.
        permutation = get_permutation(self.N_PT, self.count_swaps)
        F = self.free_energy_h(h, Ihv=self._Ihv, beta=self.betas)
        F_swapped = self.free_energy_h(
            h[permutation], Ihv=self._Ihv[permutation], beta=self.betas)
        (v, h, self._Ihv) = self.exchange_step(
            (v, h, self._Ihv), permutation, F, F_swapped)
        return (v, h)

    def exchange_step_APT(self, x):
        (v, h, z) = x
        self._Ivh = self.vlayer.compute_output(
            v, self.weights, direction='up', out=self._Ivh)
        self._Ivz = self.vlayer.compute_output(
            v, self.weights_MoI, direction='up', out=self._Ivz)
        # 0<->1, 2<->3,... if count even, 0<->0, 1<->2,3<->4,... if count odd.
        permutation = get_permutation(self.N_PT, self.count_swaps)
        F = self.free_energy_APT(
            v, Ivz=self._Ivz, Ivh=self._Ivh, beta=self.betas)
        F_swapped = self.free_energy_APT(
            v[permutation], Ivz=self._Ivz[permutation], Ivh=self._Ivh[permutation], beta=self.betas)
        (v, h, z, self._Ivh, self._Ivz) = self.exchange_step(
            (v, h, z, self._Ivh, self._Ivz), permutation, F, F_swapped)
        return (v, h, z)

    def exchange_step_APTh(self, x):
        (v, h, z) = x
        self._Ihv = self.hlayer.compute_output(
            h, self.weights, direction='down', out=self._Ihv)
        self._Ihz = self.hlayer.compute_output(
            h, self.weights_MoI, direction='up', out=self._Ihz)
        # 0<->1, 2<->3,... if count even, 0<->0, 1<->2,3<->4,... if count odd.
        permutation = get_permutation(self.N_PT, self.count_swaps)
        F = self.free_energy_APTh(
            h, Ihz=self._Ihz, Ihv=self._Ihv, beta=self.betas)
        F_swapped = self.free_energy_APTh(
            h[permutation], Ihz=self._Ihz[permutation], Ihv=self._Ihv[permutation], beta=self.betas)
        (v, h, z, self._Ihv, self._Ihz) = self.exchange_step(
            (v, h, z, self._Ihv, self._Ihz), permutation, F, F_swapped)
        return (v, h, z)

    def input_hiddens(self, v):
        if v.ndim == 1:
            v = v[np.newaxis, :]
        return self.vlayer.compute_output(v, self.weights, direction='up')

    def mean_hiddens(self, v):
        if v.ndim == 1:
            v = v[np.newaxis, :]
        return self.hlayer.mean_from_inputs(self.vlayer.compute_output(v, self.weights, direction='up'))

    def mean_visibles(self, h):
        if h.ndim == 1:
            h = h[np.newaxis, :]
        return self.vlayer.mean_from_inputs(self.hlayer.compute_output(h, self.weights, direction='down'))

    def mean_zlayer(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if self.from_MoI:
            return self.zlayer.mean_from_inputs(None, self.vlayer.compute_output(x, self.weights_MoI, direction='up'), beta=0)[:, 0]
        elif self.from_MoI_h:
            return self.zlayer.mean_from_inputs(None, self.hlayer.compute_output(x, self.weights_MoI, direction='up'), beta=0)[:, 0]
        else:
            print('No zlayer')
            return

    def likelihood_mixture(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if self.from_MoI:
            logZ = logsumexp(self.zlayer.fields0[0] + self.vlayer.logpartition(None, I0=np.swapaxes(
                self.weights_MoI[0], 0, 1), beta=0)) + self.hlayer.logpartition(None, beta=0)
            return -self.free_energy_APT(x, beta=0) - logZ
        elif self.from_MoI_h:
            logZ = logsumexp(self.zlayer.fields0[0] + self.hlayer.logpartition(None, I0=np.swapaxes(
                self.weights_MoI[0], 0, 1), beta=0)) + self.vlayer.logpartition(None, beta=0)
            return -self.free_energy_APTh(x, beta=0) - logZ
        else:
            print('No zlayer')
            return

    def sample_hiddens(self, v):
        if v.ndim == 1:
            v = v[np.newaxis, :]
        return self.hlayer.sample_from_inputs(self.vlayer.compute_output(v, self.weights, direction='up'))

    def sample_visibles(self, h):
        if h.ndim == 1:
            h = h[np.newaxis, :]
        return self.vlayer.sample_from_inputs(self.hlayer.compute_output(h, self.weights, direction='down'))

    def energy(self, x, remove_init=False):
        (v, h) = x
        if v.ndim == 1:
            v = v[np.newaxis, :]
        if h.ndim == 1:
            h = h[np.newaxis, :]
        return self.vlayer.energy(v, remove_init=remove_init) + self.hlayer.energy(h, remove_init=remove_init) - utilities.bilinear_form(self.weights, h, v, c1=self.n_ch, c2=self.n_cv)

    def free_energy(self, v, Ivh=None, beta=1):
        if v.ndim == 1:
            v = v[np.newaxis, :]
        if Ivh is None:
            Ivh = self.vlayer.compute_output(v, self.weights, direction='up')
        return self.vlayer.energy(v, beta=beta, remove_init=False) - self.hlayer.logpartition(Ivh, beta=beta)

    def free_energy_h(self, h, Ihv=None, beta=1):
        if h.ndim == 1:
            h = h[np.newaxis, :]
        if Ihv is None:
            Ihv = self.hlayer.compute_output(h, self.weights, direction='down')
        return self.hlayer.energy(h, beta=beta, remove_init=False) - self.vlayer.logpartition(Ihv, beta=beta)

    def free_energy_APT(self, v, Ivh=None, Ivz=None, beta=1):
        if Ivh is None:
            Ivh = self.vlayer.compute_output(v, self.weights, direction='up')
        if Ivz is None:
            Ivz = self.vlayer.compute_output(
                v, self.weights_MoI, direction='up')

        F = self.vlayer.energy(v, beta=beta, remove_init=False)
        F -= self.zlayer.logpartition(np.zeros(Ivz.shape,
                                               dtype=curr_float), I0=Ivz, beta=beta)
        F -= self.hlayer.logpartition(Ivh, beta=beta)
        return F

    def free_energy_APTh(self, h, Ihv=None, Ihz=None, beta=1):
        if Ihv is None:
            Ihv = self.hlayer.compute_output(h, self.weights, direction='down')
        if Ihz is None:
            Ihz = self.hlayer.compute_output(
                h, self.weights_MoI, direction='up')

        F = self.hlayer.energy(h, beta=beta, remove_init=False)
        F -= self.zlayer.logpartition(np.zeros(Ihz.shape,
                                               dtype=curr_float), I0=Ihz, beta=beta)
        F -= self.vlayer.logpartition(Ihv, beta=beta)
        return F

    # Compute all moments for RBMs with small number of hidden units.
    def compute_all_moments(self, from_hidden=True):
        if self.hidden in ['ReLU', 'Gaussian', 'ReLU+', 'dReLU']:
            from_hidden = False

        if from_hidden:
            configurations = utilities.make_all_discrete_configs(
                self.n_h, self.hidden, c=self.n_ch)
            weights = -self.free_energy_h(configurations)
            maxi = weights.max()
            weights -= maxi
            weights = np.exp(weights)
            logZ = np.log(weights.sum()) + maxi
            mean_hiddens = average(
                configurations, c=self.n_ch, weights=weights)
            mean_visibles = average(self.mean_visibles(
                configurations), weights=weights)
            covariance = average_product(configurations, self.mean_visibles(
                configurations), c1=self.n_ch, c2=self.n_cv, mean1=False, mean2=True, weights=weights)
            return logZ, mean_visibles, mean_hiddens, covariance
        else:
            configurations = utilities.make_all_discrete_configs(
                self.n_v, self.visible, c=self.n_cv)
            weights = -self.free_energy(configurations)
            maxi = weights.max()
            weights -= maxi
            weights = np.exp(weights)
            logZ = np.log(weights.sum()) + maxi
            mean_visibles = average(
                configurations, c=self.n_cv, weights=weights)
            mean_hiddens = average(self.mean_hiddens(
                configurations), weights=weights)
            covariance = average_product(self.mean_hiddens(
                configurations), configurations, c1=self.n_ch, c2=self.n_cv, mean1=True, mean2=False, weights=weights)
            return logZ, mean_visibles, mean_hiddens, covariance

    def pseudo_likelihood(self, v):
        if self.visible not in ['Bernoulli', 'Spin', 'Potts', 'Bernoulli_coupled', 'Spin_coupled', 'Potts_coupled']:
            print('PL not supported for continuous data')
        else:
            if self.visible == 'Bernoulli':
                ind = (np.arange(v.shape[0]), self.random_state.randint(
                    0, self.n_v, v.shape[0]))
                v_ = v.copy()
                v_[ind] = 1 - v[ind]
                fe = self.free_energy(v)
                fe_ = self.free_energy(v_)
                return log_logistic(fe_ - fe)
            elif self.visible == 'Spin':
                ind = (np.arange(v.shape[0]), self.random_state.randint(
                    0, self.n_v, v.shape[0]))
                v_ = v.copy()
                v_[ind] = - v[ind]
                fe = self.free_energy(v)
                fe_ = self.free_energy(v_)
                return log_logistic(fe_ - fe)
            elif self.visible == 'Potts':
                config = v
                ind_x = np.arange(config.shape[0])
                ind_y = self.random_state.randint(0, self.n_v, config.shape[0])
                E_vlayer_ref = self.vlayer.energy(
                    config) + self.vlayer.fields[ind_y, config[ind_x, ind_y]]
                output_ref = self.vlayer.compute_output(
                    config, self.weights) - self.weights[:, ind_y, config[ind_x, ind_y]].T
                fe = np.zeros([config.shape[0], self.n_cv], dtype=curr_float)
                for c in range(self.n_cv):
                    output = output_ref + self.weights[:, ind_y, c].T
                    E_vlayer = E_vlayer_ref - self.vlayer.fields[ind_y, c]
                    fe[:, c] = E_vlayer - self.hlayer.logpartition(output)
                return - fe[ind_x, config[ind_x, ind_y]] - logsumexp(- fe, 1)

    def fit(self, data, batch_size=100, learning_rate=None, extra_params=None, init='independent', optimizer='ADAM', batch_norm=True, CD=False, N_PT=1, N_MC=1, nchains=None, n_iter=10, MoI=0, MoI_h=0, MoI_tau=None,
            PTv=False, PTh=False, interpolate_z=True, degree_interpolate_z=5, zero_track_RBM=False, only_sampling=False,
            lr_decay=True, lr_final=None, decay_after=0.5, l1=0, l1b=0, l1c=0, l2=0, l2_fields=0, reg_delta=0, no_fields=False, weights=None, adapt_PT=False, AR_min=0.3, adapt_MC=False, tau_max=5, update_every=100,
            N_PT_max=20, N_MC_max=20, from_hidden=None, learning_rate_multiplier=1,
            update_betas=None, record_acceptance=None, shuffle_data=True, epsilon=1e-6, verbose=1, vverbose=0, record=[], record_interval=100, data_test=None, weights_test=None, l1_custom=None, l1b_custom=None, M_AIS=10, n_betas_AIS=10000, decay_style='geometric'):

        self.batch_size = batch_size
        self.optimizer = optimizer
        self.batch_norm = batch_norm
        self.hlayer.batch_norm = batch_norm
        self.record_swaps = False
        self.only_sampling = only_sampling

        self.n_iter = n_iter
        if self.n_iter <= 1:
            lr_decay = False

        if learning_rate is None:
            if self.optimizer in ['SGD', 'momentum']:
                if self.hidden in ['Gaussian', 'ReLU+', 'ReLU', 'dReLU']:
                    if self.batch_norm:
                        learning_rate = 0.05
                    else:
                        learning_rate = 5e-3
                else:
                    learning_rate = 0.05

            elif self.optimizer == 'ADAM':
                learning_rate = 5e-3
            else:
                print('Need to specify learning rate for optimizer.')

        self.learning_rate_init = copy.copy(learning_rate)
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        if self.lr_decay:
            self.decay_after = decay_after
            self.start_decay = int(self.n_iter * self.decay_after)
            if lr_final is None:
                self.lr_final = 1e-2 * self.learning_rate
            else:
                self.lr_final = lr_final
            self.decay_gamma = (float(self.lr_final) / float(self.learning_rate)
                                )**(1 / float(self.n_iter * (1 - self.decay_after)))
        else:
            self.decay_gamma = 1

        self.no_fields = no_fields

        data = np.asarray(data, dtype=self.vlayer.type, order="c")
        if weights is not None:
            weights = np.asarray(weights, dtype=curr_float)
        if self.batch_norm:
            self.mu_data = utilities.average(
                data, c=self.n_cv, weights=weights)
        self.moments_data = self.vlayer.get_moments(
            data,  value='data', weights=weights, beta=1)

        n_samples = data.shape[0]
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))

        if self.only_sampling:
            init = 'previous'
            for layer_ in ['vlayer', 'hlayer']:
                for param in self.__dict__[layer_].list_params:
                    self.__dict__[layer_].__dict__[
                        param + '0'] = np.zeros(self.__dict__[layer_].__dict__[param + '0'].shape, dtype=curr_float)
                    if 'gamma' in param:
                        self.__dict__[layer_].__dict__[param + '0'] += 1
                    if self.interpolate:
                        self.__dict__[layer_].__dict__[
                            param + '1'] = np.zeros(self.__dict__[layer_].__dict__[param + '1'].shape, dtype=curr_float)

        if init != 'previous':
            norm_init = np.sqrt(0.1 / self.n_v)

            self.init_weights(norm_init)
            if init == 'independent':
                self.vlayer.init_params_from_data(
                    self.moments_data, eps=epsilon, value='moments')
            self.hlayer.init_params_from_data(None)

        if (bool(MoI) | bool(MoI_h)) & (N_PT == 1):
            adapt_PT = True

        self.adapt_PT = adapt_PT
        self.AR_min = AR_min
        self.N_PT_max = N_PT_max
        if self.adapt_PT:
            N_PT = 2
            update_betas = True
            record_acceptance = True

        self.adapt_MC = adapt_MC
        self.tau_max = tau_max
        self.N_MC_max = N_MC_max
        if self.adapt_MC:
            N_MC = 1

        self.N_PT = N_PT
        self.N_MC = N_MC
        if from_hidden is None:
            if self.visible == 'Bernoulli':
                # Data is sparse.
                if np.minimum(self.mu_data, 1 - self.mu_data).mean() < 0.2:
                    from_hidden = True
                else:
                    from_hidden = False
            elif self.n_cv > 4:  # Amino acids; but not for RNA.
                from_hidden = True
            else:
                from_hidden = False

        self.from_hidden = from_hidden
        self.zero_track_RBM = zero_track_RBM

        if N_MC == 0:
            if self.from_hidden:
                if self.n_ch > 1:
                    nchains = (self.n_ch)**self.n_h
                else:
                    nchains = 2**self.n_h
            else:
                if self.n_cv > 1:
                    nchains = (self.n_cv)**self.n_v
                else:
                    nchains = 2**self.n_v
        else:
            self.from_hidden = False

        if nchains is None:
            self.nchains = self.batch_size
        else:
            self.nchains = nchains

        self.CD = CD
        self.l1 = l1
        self.l1b = l1b
        self.l1c = l1c
        self.l1_custom = l1_custom
        self.l1b_custom = l1b_custom
        self.l2 = l2
        self.tmp_l2_fields = l2_fields
        self.tmp_reg_delta = reg_delta

        if self.N_PT > 1:
            if record_acceptance == None:
                record_acceptance = True
            self.record_acceptance = record_acceptance

            if update_betas == None:
                update_betas = True

            self._update_betas = update_betas

            if self.record_acceptance:
                self.mavar_gamma = 0.9
                self.acceptance_rates = np.zeros(N_PT - 1, dtype=curr_float)
                self.log_acceptance_rates = np.ones(N_PT - 1, dtype=curr_float)
                self.mav_acceptance_rates = np.zeros(
                    N_PT - 1, dtype=curr_float)
                self.mav_log_acceptance_rates = np.ones(
                    N_PT - 1, dtype=curr_float)
            self.count_swaps = 0

            if self._update_betas:
                record_acceptance = True
                self.update_betas_lr = 0.025
                self.update_betas_lr_decay = self.decay_gamma

            if self._update_betas | (not hasattr(self, 'betas')):
                self.betas = np.arange(N_PT) / (N_PT - 1)
                self.betas = self.betas[::-1].astype(curr_float)
            if (len(self.betas) != N_PT):
                self.betas = np.arange(N_PT) / (N_PT - 1)
                self.betas = self.betas[::-1].astype(curr_float)

            if bool(MoI) | bool(MoI_h):
                if MoI:
                    self.from_MoI = True
                    self.from_MoI_h = False
                    if type(MoI) == int:
                        MoI = moi.MoI(N=self.n_v, M=MoI,
                                      nature=self.visible, n_c=self.n_cv)
                        if verbose | vverbose:
                            print('Fitting MOI first')
                        MoI.fit(data, weights=weights, verbose=verbose)
                        if verbose | vverbose:
                            print('Fitting MOI done')
                else:
                    self.from_MoI = False
                    self.from_MoI_h = True
                self.PTv = False
                self.PTh = False
                self.interpolate_z = interpolate_z
                if self.interpolate_z:
                    if degree_interpolate_z is None:
                        degree_interpolate_z = 5
                if self.from_MoI:
                    self.init_zlayer_APT(
                        MoI, interpolate=self.interpolate_z, degree_interpolate=degree_interpolate_z, layer_id=0)
                else:
                    self.init_zlayer_APT(MoI_h, interpolate=self.interpolate_z,
                                         degree_interpolate=degree_interpolate_z, layer_id=1)

                if (self.from_MoI & self.zero_track_RBM) | self.from_MoI_h:
                    self.update_zlayer_mu_lr = 0.25
            else:
                self.from_MoI = False
                self.from_MoI_h = False
                self.interpolate_z = False
                self.PTv = PTv
                self.PTh = PTh
        else:
            self.from_MoI = False
            self.from_MoI_h = False
            self.interpolate_z = False
            self._update_betas = False
            self.PTv = False
            self.PTh = False

        self.hlayer.update0 = True
        if self.zero_track_RBM:
            self.vlayer.update0 = True
            self.hlayer.target0 = 'neg'
            self.vlayer.target0 = 'neg'
        else:
            self.vlayer.update0 = False
            self.hlayer.target0 = 'pos'
            self.vlayer.target0 = 'pos'

        if self.adapt_PT:
            list_attr_adapt_PT = ['acceptance_rates', 'mav_acceptance_rates', 'log_acceptance_rates',
                                  'mav_log_acceptance_rates', 'fantasy_v', 'fantasy_h', '_Ivh', '_Ihv']
            if self.from_MoI:
                list_attr_adapt_PT += ['fantasy_z', '_Ivz', '_Izv']
            elif self.from_MoI_h:
                list_attr_adapt_PT += ['fantasy_z', '_Ihz', '_Izh']
            else:
                list_attr_adapt_PT += ['fantasy_E']

        if self.N_PT > 1:
            self.fantasy_v = self.vlayer.random_init_config(
                self.nchains * self.N_PT).reshape([self.N_PT, self.nchains, self.vlayer.N])
            self.fantasy_h = self.hlayer.random_init_config(
                self.nchains * self.N_PT).reshape([self.N_PT, self.nchains, self.hlayer.N])
            if self.from_MoI | self.from_MoI_h:
                self.fantasy_z = self.zlayer.random_init_config(
                    self.nchains * self.N_PT).reshape([self.N_PT, self.nchains, 1])
            else:
                self.fantasy_E = np.zeros(
                    [self.N_PT, self.nchains], dtype=curr_float)
        else:
            if self.N_MC == 0:
                if self.from_hidden:
                    self.fantasy_h = utilities.make_all_discrete_configs(
                        self.n_h, self.hidden, c=self.n_ch)
                    # self.fantasy_h = self.hlayer.random_init_config(self.nchains)
                else:
                    self.fantasy_v = utilities.make_all_discrete_configs(
                        self.n_v, self.visible, c=self.n_cv)
            else:
                self.fantasy_v = self.vlayer.random_init_config(self.nchains)
                self.fantasy_h = self.hlayer.random_init_config(self.nchains)

        self.gradient = self.initialize_gradient_dictionary()
        self.do_grad_updates = {
            'vlayer': self.vlayer.do_grad_updates, 'weights': True}
        if self.batch_norm:
            self.do_grad_updates['hlayer'] = self.hlayer.do_grad_updates_batch_norm
        else:
            self.do_grad_updates['hlayer'] = self.hlayer.do_grad_updates
        if (self.from_MoI | self.from_MoI_h):
            self.do_grad_updates['zlayer'] = self.zlayer.do_grad_updates
        if (self.from_MoI & self.zero_track_RBM) | self.from_MoI_h:
            self.do_grad_updates['weights_MoI'] = True
        if self.only_sampling:
            self.batch_norm = False
            for layer_ in ['vlayer', 'hlayer']:
                for key in self.do_grad_updates[layer_]:
                    if not ('0' in key) | ('1' in key):
                        self.do_grad_updates[layer_][key] = False
            self.do_grad_updates['weights'] = False

        if self.optimizer == 'momentum':
            if extra_params is None:
                extra_params = 0.9
            self.momentum = extra_params
            self.previous_update = self.initialize_gradient_dictionary()

        elif self.optimizer == 'ADAM':
            if extra_params is None:
                extra_params = [0.99, 0.99, 1e-3]
            self.beta1 = extra_params[0]
            self.beta2 = extra_params[1]
            self.epsilon = extra_params[2]

            self.gradient_moment1 = self.initialize_gradient_dictionary()
            self.gradient_moment2 = self.initialize_gradient_dictionary()

        self.learning_rate_multiplier = {}
        for key in self.gradient.keys():
            if type(self.gradient[key]) == dict:
                self.learning_rate_multiplier[key] = {}
                for key_ in self.gradient[key].keys():
                    if ('0' in key_) | ('1' in key_):
                        self.learning_rate_multiplier[key][key_] = learning_rate_multiplier
                    else:
                        self.learning_rate_multiplier[key][key_] = 1
            else:
                if key == 'weights_MoI':
                    self.learning_rate_multiplier[key] = learning_rate_multiplier
                else:
                    self.learning_rate_multiplier[key] = 1

        self.has_momentum = {}
        for key in self.gradient.keys():
            if type(self.gradient[key]) == dict:
                self.has_momentum[key] = {}
                for key_ in self.gradient[key].keys():
                    if ('0' in key_) and not self.zero_track_RBM:
                        self.has_momentum[key][key_] = True
                    else:
                        self.has_momentum[key][key_] = False
            else:
                if key == 'weights_MoI':
                    self.has_momentum[key] = ~self.zero_track_RBM
                else:
                    self.has_momentum[key] = False

        if ('TAU' in record) | self.adapt_MC:
            if self.N_PT > 1:
                if self.from_MoI_h:
                    x = self.fantasy_h[0]
                else:
                    x = self.fantasy_v[0]
            else:
                x = self.fantasy_v
            if MoI_tau is not None:
                previous_z = np.asarray(np.argmax(MoI_tau.expectation(
                    x), axis=-1)[:, np.newaxis], order='c', dtype=curr_int)
                n_z = MoI_tau.M

            else:
                previous_z = np.asarray(np.argmax(self.mean_zlayer(
                    x), axis=-1)[:, np.newaxis], order='c', dtype=curr_int)
                n_z = self.zlayer.n_c
            joint_z = np.zeros([n_z, n_z], dtype=curr_float)
            smooth_jointz = 0.01**(1.0 / record_interval)
            tau = 0

        if shuffle_data:
            if weights is not None:
                permute = np.arange(data.shape[0])
                self.random_state.shuffle(permute)
                weights = weights[permute]
                data = data[permute, :]
            else:
                self.random_state.shuffle(data)

        if weights is not None:
            weights /= weights.mean()
        self.count_updates = 0
        if verbose:
            if weights is not None:
                lik = (self.pseudo_likelihood(data) *
                       weights).sum() / weights.sum()
            else:
                lik = self.pseudo_likelihood(data).mean()
            print('Iteration number 0, pseudo-likelihood: %.2f' % lik)

        result = {}
        for key in record:
            result[key] = []

        count = 0

        for epoch in range(1, n_iter + 1):
            if verbose:
                begin = time.time()
            if self.lr_decay:
                if (epoch > self.start_decay):
                    self.learning_rate *= self.decay_gamma
                    if self._update_betas:
                        self.update_betas_lr *= self.decay_gamma
                    if (self.from_MoI & self.zero_track_RBM) | self.from_MoI_h:
                        self.update_zlayer_mu_lr *= self.decay_gamma

            if (verbose | vverbose):
                print('Starting epoch %s' % (epoch))
            for batch_slice in batch_slices:
                if weights is None:
                    no_nans = self.minibatch_fit(
                        data[batch_slice], weights=None, verbose=vverbose)
                else:
                    no_nans = self.minibatch_fit(
                        data[batch_slice], weights=weights[batch_slice], verbose=vverbose)

                if ('TAU' in record) | self.adapt_MC:
                    if self.N_PT > 1:
                        if self.from_MoI_h:
                            x = self.fantasy_h[0]
                        else:
                            x = self.fantasy_v[0]
                    else:
                        x = self.fantasy_v
                    if MoI_tau is not None:
                        current_z = np.asarray(np.argmax(MoI_tau.expectation(
                            x), axis=-1)[:, np.newaxis], order='c', dtype=curr_int)
                    else:
                        current_z = np.asarray(np.argmax(self.mean_zlayer(
                            x), axis=-1)[:, np.newaxis], order='c', dtype=curr_int)

                    joint_z = (1 - smooth_jointz) * utilities.average_product(previous_z,
                                                                              current_z, c1=n_z, c2=n_z)[0, 0] + smooth_jointz * joint_z
                    previous_z = current_z.copy()

                if self.adapt_PT:
                    if (count > 0) & (count % update_every == 0):
                        curr_AR = self.mav_acceptance_rates.mean() / (1 - self.mavar_gamma**(1 + count))
                        if (curr_AR < self.AR_min) & (self.N_PT < self.N_PT_max):
                            self.N_PT += 1
                            self.betas = np.concatenate(
                                (self.betas * (self.N_PT - 2) / (self.N_PT - 1) + 1.0 / (self.N_PT - 1), self.betas[-1:]))
                            for attr in list_attr_adapt_PT:
                                previous = getattr(self, attr)
                                new = np.concatenate(
                                    (previous, previous[-1:]), axis=0)
                                setattr(self, attr,  new)
                            if verbose | vverbose:
                                print('AR = %.3f, Increasing N_PT to %s' %
                                      (curr_AR, self.N_PT))

                        elif (curr_AR > 1.25 * (self.N_PT / (self.N_PT - 1)) * self.AR_min) & (self.N_PT > 2):
                            self.N_PT -= 1
                            self.betas = (
                                self.betas[:-1] - self.betas[-2]) / (1 - self.betas[-2])
                            for attr in list_attr_adapt_PT:
                                previous = getattr(self, attr)
                                new = previous[:-1]
                                setattr(self, attr,  new)

                            if verbose | vverbose:
                                print('AR = %.3f, Decreasing N_PT to %s' %
                                      (curr_AR, self.N_PT))

                if self.adapt_MC:
                    if (count > 0) & (count % update_every == 0):
                        Q = joint_z / (joint_z.sum(0) + 1e-10)[np.newaxis, :]
                        lam, v = np.linalg.eig(Q)
                        lam = lam[np.argsort(np.abs(np.real(lam)))[::-1]]
                        tau = -1 / np.log(np.abs(np.real(lam[1])))
                        self.tau = tau
                        if (tau > self.tau_max) & (self.N_MC < self.N_MC_max):
                            self.N_MC += 1
                            if verbose | vverbose:
                                print('tau = %.2f, Increasing N_MC to %s' %
                                      (tau, self.N_MC))
                        elif (tau < (self.N_MC - 1) / float(self.N_MC) * self.tau_max) & (self.N_MC > 1):
                            self.N_MC -= 1
                            if verbose | vverbose:
                                print('tau = %.2f, Decreasing N_MC to %s' %
                                      (tau, self.N_MC))

                if (count % record_interval == 0):
                    for key in record:
                        if key == 'PL':
                            result['PL'].append(utilities.average(
                                self.pseudo_likelihood(data), weights=weights))
                        elif key == 'PL_test':
                            result['PL_test'].append(utilities.average(
                                self.pseudo_likelihood(data_test), weights=weights_test))
                        elif key == 'L':
                            if M_AIS > 0:
                                N_PT = copy.copy(self.N_PT)
                                self.AIS(M=M_AIS, n_betas=n_betas_AIS,
                                         verbose=0, beta_type='linear')
                                self.N_PT = N_PT
                            else:
                                logZ, _, _, _ = self.compute_all_moments(
                                    from_hidden=self.from_hidden)
                                self.log_Z_AIS = logZ
                            result['L'].append(utilities.average(
                                self.likelihood(data, recompute_Z=False), weights=weights))
                        elif key == 'L_test':
                            result['L_test'].append(utilities.average(self.likelihood(
                                data_test, recompute_Z=False), weights=weights_test))
                        elif key == 'L_mixture':
                            if self.from_MoI:
                                L_mixture = self.likelihood_mixture(data)
                            elif self.from_MoI_h:
                                L_mixture = self.likelihood_mixture(
                                    self.sample_hiddens(data))
                            else:
                                print('not supported')
                            result['L_mixture'].append(
                                utilities.average(L_mixture, weights=weights))
                        elif key == 'norm':
                            if self.n_cv > 1:
                                result['norm'].append(
                                    (self.weights**2).sum(-1).sum(-1))
                            else:
                                result['norm'].append(
                                    (self.weights**2).sum(-1))
                        elif key == 'p':
                            if self.n_cv > 1:
                                tmp = (self.weights**2).sum(-1)
                            else:
                                tmp = (self.weights**2)
                            a = 3
                            result['p'].append(
                                (tmp**a).sum(-1)**2 / (tmp**(2 * a)).sum(-1) / self.n_v)
                        elif key == 'TAU':
                            Q = joint_z / (joint_z.sum(0) +
                                           1e-10)[np.newaxis, :]
                            lam, v = np.linalg.eig(Q)
                            lam = lam[np.argsort(np.abs(np.real(lam)))[::-1]]
                            tau = -1 / np.log(np.abs(np.real(lam[1])))
                            result['TAU'].append(tau.copy())
                        elif key == 'AR':
                            result['AR'].append(
                                self.mav_acceptance_rates.mean())
                        elif key == 'mu_fantasy_v':
                            result[key].append(self.fantasy_v.mean(-1))
                        else:
                            if key[:2] == 'v_':
                                result[key].append(
                                    self.vlayer.__dict__[key[2:]].copy())
                            elif key[:2] == 'h_':
                                result[key].append(
                                    self.hlayer.__dict__[key[2:]].copy())
                            elif key[:2] == 'z_':
                                result[key].append(
                                    self.zlayer.__dict__[key[2:]].copy())
                            else:
                                result[key].append(self.__dict__[key].copy())
                count += 1
                if not no_nans:
                    done = True
                    break
                else:
                    done = False

            if done:
                break

            if verbose:
                end = time.time()
                if weights is not None:
                    lik = (self.pseudo_likelihood(data) *
                           weights).sum() / weights.sum()
                else:
                    lik = self.pseudo_likelihood(data).mean()
                message = "[%s] Iteration %d, time = %.2fs, pseudo-likelihood = %.2f" % (
                    type(self).__name__, epoch, end - begin, lik)
                if self.N_PT > 1:
                    AR = self.mav_acceptance_rates.mean()
                    message += ", AR = %.3f" % AR
                if ('TAU' in record) | self.adapt_MC:
                    message += ", tau = %.2f" % tau
                print(message)

            if shuffle_data:
                if weights is not None:
                    permute = np.arange(data.shape[0])
                    self.random_state.shuffle(permute)
                    weights = weights[permute]
                    data = data[permute, :]
                else:
                    self.random_state.shuffle(data)

        for key, item in result.items():
            if key == 'betas':
                max_len = max([len(x) for x in item])
                len_training = len(item)
                result[key] = np.zeros([len_training, max_len])
                for i, beta_ in enumerate(item):
                    result[key][i, :len(beta_)] = np.array(beta_)
            else:
                result[key] = np.array(item)
        return result

    def minibatch_fit(self, V_pos, weights=None, verbose=True):
        self.count_updates += 1
        if self.CD:  # Contrastive divergence: initialize the Markov chain at the data point.
            self.fantasy_v = V_pos
        # Else: use previous value.
        for _ in range(self.N_MC):
            if self.N_PT > 1:
                if self.from_MoI:
                    (self.fantasy_v, self.fantasy_h, self.fantasy_z) = self.exchange_step_APT(
                        (self.fantasy_v, self.fantasy_h, self.fantasy_z))
                    (self.fantasy_v, self.fantasy_h, self.fantasy_z) = self.markov_step_APT(
                        (self.fantasy_v, self.fantasy_h, self.fantasy_z), beta=self.betas, recompute=False)
                elif self.from_MoI_h:
                    (self.fantasy_v, self.fantasy_h, self.fantasy_z) = self.exchange_step_APTh(
                        (self.fantasy_v, self.fantasy_h, self.fantasy_z))
                    (self.fantasy_v, self.fantasy_h, self.fantasy_z) = self.markov_step_APTh(
                        (self.fantasy_v, self.fantasy_h, self.fantasy_z), beta=self.betas, recompute=False)
                elif self.PTv:
                    (self.fantasy_v, self.fantasy_h) = self.exchange_step_PTv(
                        (self.fantasy_v, self.fantasy_h))
                    (self.fantasy_v, self.fantasy_h) = self.markov_step(
                        (self.fantasy_v, self.fantasy_h), beta=self.betas, recompute=False)
                elif self.PTh:
                    (self.fantasy_v, self.fantasy_h) = self.exchange_step_PTh(
                        (self.fantasy_v, self.fantasy_h))
                    (self.fantasy_v, self.fantasy_h) = self.markov_step_h(
                        (self.fantasy_v, self.fantasy_h), beta=self.betas, recompute=False)
                else:
                    (self.fantasy_v, self.fantasy_h), self.fantasy_E = self.markov_step_and_energy(
                        (self.fantasy_v, self.fantasy_h), self.fantasy_E, beta=self.betas)
                    (self.fantasy_v, self.fantasy_h), self.fantasy_E = self.exchange_step_PT(
                        (self.fantasy_v, self.fantasy_h), self.fantasy_E, compute_energy=False)
            else:
                (self.fantasy_v, self.fantasy_h) = self.markov_step(
                    (self.fantasy_v, self.fantasy_h))

        for attr in ['fantasy_v', 'fantasy_h', 'fantasy_z', 'fantasy_E']:
            if hasattr(self, attr):
                if np.isnan(getattr(self, attr)).max():
                    print('NAN in %s (before gradient computation). Breaking' % attr)
                    return False

        if self.N_MC > 0:  # Regular Monte Carlo
            weights_neg = None
        # No Monte Carlo. Compute exhaustively the moments using all 2**N configurations.
        else:
            if self.from_hidden:
                F = self.free_energy_h(self.fantasy_h)
            else:
                F = self.free_energy(self.fantasy_v)
            F -= F.min()
            weights_neg = np.exp(-F)
            weights_neg /= weights_neg.sum()

        if self.from_hidden:
            if self.N_PT > 1:
                H_neg = self.fantasy_h[0]
            else:
                H_neg = self.fantasy_h
            I_neg = self.hlayer.compute_out(
                H_neg, self.weights, direction='down')
        else:
            if self.N_PT > 1:
                V_neg = self.fantasy_v[0]
            else:
                V_neg = self.fantasy_v
            I_neg = self.vlayer.compute_output(V_neg, self.weights)

        I_pos = self.vlayer.compute_output(V_pos, self.weights)

        if self.batch_norm:
            if (self.n_cv > 1) & (self.n_ch == 1):
                mu_I = np.tensordot(
                    self.weights, self.mu_data, axes=[(1, 2), (0, 1)])
            elif (self.n_cv > 1) & (self.n_ch > 1):
                mu_I = np.tensordot(
                    self.weights, self.mu_data, axes=[(1, 3), (0, 1)])
            elif (self.n_cv == 1) & (self.n_ch > 1):
                mu_I = np.tensordot(self.weights, self.mu_data, axes=[1, 0])
            else:
                mu_I = np.dot(self.weights, self.mu_data)

            self.hlayer.batch_norm_update(
                mu_I, I_pos, lr=0.25 * self.learning_rate / self.learning_rate_init, weights=weights)

        H_pos = self.hlayer.mean_from_inputs(I_pos)

        if (self.from_MoI & self.zero_track_RBM):
            data_0v = np.swapaxes(self.weights_MoI[0], 0, 1)
            weights_0v = self.zlayer.mu[0]
            data_0h = None
            weights_0h = None
        elif self.from_MoI_h:
            data_0v = None
            weights_0v = None
            data_0h = np.swapaxes(self.weights_MoI[0], 0, 1)
            weights_0h = self.zlayer.mu[0]
        else:
            data_0v = None
            data_0h = None
            weights_0v = None
            weights_0h = None

        if self.from_MoI & self.zero_track_RBM:
            Z = self.zlayer.mean_from_inputs(
                None, I0=self.vlayer.compute_output(V_neg, self.weights_MoI), beta=0)
            self.gradient['weights_MoI'] = utilities.average_product(
                Z, V_neg, mean1=True, mean2=False, c1=self.zlayer.n_c, c2=self.vlayer.n_c) - self.muzx
        elif self.from_MoI_h:
            if self.zero_track_RBM:
                Z = self.zlayer.mean_from_inputs(None, I0=self.hlayer.compute_output(
                    self.fantasy_h[0], self.weights_MoI), beta=0)
                self.gradient['weights_MoI'] = utilities.average_product(
                    Z, self.fantasy_h[0], mean1=True, mean2=False, c1=self.zlayer.n_c, c2=self.hlayer.n_c) - self.muzx
            else:
                H_pos_sample = self.hlayer.sample_from_inputs(I_pos)
                Z = self.zlayer.mean_from_inputs(None, I0=self.hlayer.compute_output(
                    H_pos_sample, self.weights_MoI), beta=0)
                self.gradient['weights_MoI'] = utilities.average_product(
                    Z, H_pos_sample, mean1=True, mean2=False, c1=self.zlayer.n_c, c2=self.hlayer.n_c) - self.muzx

        if self.from_hidden:
            self.gradient['vlayer'] = self.vlayer.internal_gradients(self.moments_data, I_neg, data_0=data_0v,
                                                                     weights=None, weights_neg=weights_neg, weights_0=weights_0v,
                                                                     value='moments', value_neg='input', value_0='input')

            self.gradient['hlayer'] = self.hlayer.internal_gradients(I_pos, H_neg, data_0=data_0h,
                                                                     weights=weights, weights_neg=weights_neg, weights_0=weights_0h,
                                                                     value='input', value_neg='data', value_0='input')

            V_neg = self.vlayer.mean_from_inputs(I_neg)
            self.gradient['weights'] = pgm.couplings_gradients_h(self.weights, H_pos, H_neg, V_pos, V_neg, self.n_ch, self.n_cv, l1=self.l1,
                                                                 l1b=self.l1b, l1c=self.l1c, l2=self.l2, weights=weights, weights_neg=weights_neg, l1_custom=self.l1_custom, l1b_custom=self.l1b_custom)

        else:
            self.gradient['vlayer'] = self.vlayer.internal_gradients(self.moments_data, V_neg, data_0=data_0v,
                                                                     weights=weights, weights_neg=weights_neg, weights_0=weights_0v,
                                                                     value='moments', value_neg='data', value_0='input')

            self.gradient['hlayer'] = self.hlayer.internal_gradients(I_pos, I_neg, data_0=data_0h,
                                                                     weights=weights, weights_neg=weights_neg, weights_0=weights_0h,
                                                                     value='input', value_neg='input', value_0='input')

            H_neg = self.hlayer.mean_from_inputs(I_neg)
            self.gradient['weights'] = pgm.couplings_gradients(self.weights, H_pos, H_neg, V_pos, V_neg, self.n_ch, self.n_cv, mean1=True, l1=self.l1,
                                                               l1b=self.l1b, l1c=self.l1c, l2=self.l2, weights=weights, weights_neg=weights_neg, l1_custom=self.l1_custom, l1b_custom=self.l1b_custom)

        if self.interpolate & (self.N_PT > 2):
            self.gradient['vlayer'] = self.vlayer.internal_gradients_interpolation(
                self.fantasy_v, self.betas, gradient=self.gradient['vlayer'], value='data')
            self.gradient['hlayer'] = self.hlayer.internal_gradients_interpolation(
                self.fantasy_h, self.betas, gradient=self.gradient['hlayer'], value='data')
        if self.interpolate_z & (self.N_PT > 2):
            self.gradient['zlayer'] = self.zlayer.internal_gradients_interpolation(
                self.fantasy_z, self.betas, value='data')

        if check_nan(self.gradient, what='gradient', location='before batch norm'):
            self.vproblem = V_pos
            self.Iproblem = I_pos
            return False

        if self.batch_norm:  # Modify gradients.
            self.hlayer.batch_norm_update_gradient(
                self.gradient['weights'], self.gradient['hlayer'], V_pos, I_pos, self.mu_data, self.n_cv, weights=weights)

        if check_nan(self.gradient, what='gradient', location='after batch norm'):
            self.vproblem = V_pos
            self.Iproblem = I_pos
            return False

        for key, item in self.gradient.items():
            if type(item) == dict:
                for key_, item_ in item.items():
                    saturate(item_, 1.0)
            else:
                saturate(item, 1.0)

        if self.tmp_l2_fields > 0:
            self.gradient['vlayer']['fields'] -= self.tmp_l2_fields * \
                self.vlayer.fields
        if not self.tmp_reg_delta == 0:
            self.gradient['hlayer']['delta'] -= self.tmp_reg_delta

        for key, item in self.gradient.items():
            if type(item) == dict:
                for key_, item_ in item.items():
                    current = getattr(getattr(self, key), key_)
                    do_update = self.do_grad_updates[key][key_]
                    lr_multiplier = self.learning_rate_multiplier[key][key_]
                    has_momentum = self.has_momentum[key][key_]
                    gradient = item_
                    if do_update:
                        if self.optimizer == 'SGD':
                            current += self.learning_rate * lr_multiplier * gradient
                        elif self.optimizer == 'momentum':
                            self.previous_update[key][key_] = (
                                1 - self.momentum) * self.learning_rate * lr_multiplier * gradient + self.momentum * self.previous_update[key][key_]
                            current += self.previous_update[key][key_]
                        elif self.optimizer == 'ADAM':
                            if has_momentum:
                                self.gradient_moment1[key][key_] *= self.beta1
                                self.gradient_moment1[key][key_] += (
                                    1 - self.beta1) * gradient
                                self.gradient_moment2[key][key_] *= self.beta2
                                self.gradient_moment2[key][key_] += (
                                    1 - self.beta2) * gradient**2
                                current += self.learning_rate * lr_multiplier / (1 - self.beta1) * (self.gradient_moment1[key][key_] / (
                                    1 - self.beta1**self.count_updates)) / (self.epsilon + np.sqrt(self.gradient_moment2[key][key_] / (1 - self.beta2**self.count_updates)))
                            else:
                                self.gradient_moment2[key][key_] *= self.beta2
                                self.gradient_moment2[key][key_] += (
                                    1 - self.beta2) * gradient**2
                                current += self.learning_rate * lr_multiplier * gradient / \
                                    (self.epsilon + np.sqrt(self.gradient_moment2[key][key_] / (
                                        1 - self.beta2**self.count_updates)))

                            # self.gradient_moment1[key][key_] *= self.beta1
                            # self.gradient_moment1[key][key_] += (1- self.beta1) * gradient
                            # self.gradient_moment2[key][key_] *= self.beta2
                            # self.gradient_moment2[key][key_] += (1- self.beta2) * gradient**2
                            # current += self.learning_rate * (self.gradient_moment1[key][key_]/(1-self.beta1**self.count_updates)) /(self.epsilon + np.sqrt( self.gradient_moment2[key][key_]/(1-self.beta2**self.count_updates ) ) )
            else:
                current = getattr(self, key)
                do_update = self.do_grad_updates[key]
                lr_multiplier = self.learning_rate_multiplier[key]
                has_momentum = self.has_momentum[key]
                gradient = item
                if do_update:
                    if self.optimizer == 'SGD':
                        current += self.learning_rate * lr_multiplier * gradient
                    elif self.optimizer == 'momentum':
                        self.previous_update[key] = (
                            1 - self.momentum) * self.learning_rate * lr_multiplier * gradient + self.momentum * self.previous_update[key]
                        current += self.previous_update[key]
                    elif self.optimizer == 'ADAM':
                        if has_momentum:
                            self.gradient_moment1[key] *= self.beta1
                            self.gradient_moment1[key] += (
                                1 - self.beta1) * gradient
                            self.gradient_moment2[key] *= self.beta2
                            self.gradient_moment2[key] += (
                                1 - self.beta2) * gradient**2
                            current += self.learning_rate * lr_multiplier / (1 - self.beta1) * (self.gradient_moment1[key] / (
                                1 - self.beta1**self.count_updates)) / (self.epsilon + np.sqrt(self.gradient_moment2[key] / (1 - self.beta2**self.count_updates)))
                        else:
                            self.gradient_moment2[key] *= self.beta2
                            self.gradient_moment2[key] += (
                                1 - self.beta2) * gradient**2
                            current += self.learning_rate * lr_multiplier * gradient / \
                                (self.epsilon + np.sqrt(self.gradient_moment2[key] / (
                                    1 - self.beta2**self.count_updates)))

                        # self.gradient_moment1[key] *= self.beta1
                        # self.gradient_moment1[key] += (1- self.beta1) * gradient
                        # self.gradient_moment2[key] *= self.beta2
                        # self.gradient_moment2[key] += (1- self.beta2) * gradient**2
                        # current += self.learning_rate * (self.gradient_moment1[key]/(1-self.beta1**self.count_updates)) /(self.epsilon + np.sqrt( self.gradient_moment2[key]/(1-self.beta2**self.count_updates ) ) )

        if (self.n_cv > 1) | (self.n_ch > 1):
            pgm.gauge_adjust_couplings(
                self.weights, self.n_ch, self.n_cv, gauge=self.gauge)

        self.hlayer.recompute_params()
        self.vlayer.ensure_constraints()
        self.hlayer.ensure_constraints()

        if check_nan(self.hlayer.__dict__, what='hlayer', location='after recompute parameters'):
            return False

        if (self.from_MoI & self.zero_track_RBM) | self.from_MoI_h:
            if self.from_MoI:
                layer_id = 0
                n_cx = self.vlayer.n_c
                # fantasy_x = self.fantasy_v[0]
            else:
                layer_id = 1
                n_cx = self.hlayer.n_c
                # if self.zero_track_RBM:
                # fantasy_x = self.fantasy_h[0]
                # else:
                # fantasy_x = H_pos_sample

            if self.zero_track_RBM:
                weights_Z = weights_neg
            else:
                weights_Z = weights

            pgm.gauge_adjust_couplings(
                self.weights_MoI, self.zlayer.n_c, n_cx, gauge=self.gauge)
            # muz = utilities.average(Z,weights=weights_Z)
            # if weights_Z is None:
            #     likz =  np.dot(self.likelihood_mixture(fantasy_x),Z[:,0])/(muz[0] * fantasy_x.shape[0] )
            # else:
            #     likz =  np.dot(self.likelihood_mixture(fantasy_x) * weights_Z,Z[:,0])/(muz[0] * fantasy_x.shape[0] )

            self.zlayer.mu = (1 - self.update_zlayer_mu_lr) * self.zlayer.mu + \
                self.update_zlayer_mu_lr * \
                utilities.average(Z, weights=weights_Z)
            # self.zlayer.average_likelihood = 0. * self.zlayer.average_likelihood + 1 * likz
            self.update_params_MoI(
                layer_id=layer_id, eps=1e-4, verbose=verbose)

        if self.N_PT > 1:
            if self._update_betas:
                self.update_betas()

        return True

    def initialize_gradient_dictionary(self):
        out = {}
        out['vlayer'] = self.vlayer.internal_gradients(
            None, None, value='input', value_neg='input')
        out['hlayer'] = self.hlayer.internal_gradients(
            None, None, value='input', value_neg='input')
        for layer_ in ['vlayer', 'hlayer']:
            for key, item in out[layer_].items():
                item *= 0
        out['weights'] = np.zeros_like(self.weights)
        if (self.from_MoI & self.zero_track_RBM) | self.from_MoI_h:
            out['weights_MoI'] = np.zeros_like(self.weights_MoI)
        if self.interpolate:
            for attr in self.vlayer.list_params:
                out['vlayer'][attr +
                              '1'] = np.zeros_like(getattr(self.vlayer, attr + '1'))
            for attr in self.hlayer.list_params:
                out['hlayer'][attr +
                              '1'] = np.zeros_like(getattr(self.hlayer, attr + '1'))
        if self.interpolate_z:
            out['zlayer'] = {'fields1': np.zeros_like(self.zlayer.fields1)}
        return out
