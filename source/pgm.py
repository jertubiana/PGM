
# -*- coding: utf-8 -*-
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
import utilities
import copy
from scipy.sparse import diags
import time
import layer
from float_precision import double_precision, curr_float, curr_int


def couplings_gradients(W, X1_p, X1_n, X2_p, X2_n, n_c1, n_c2, mean1=False, mean2=False, l1=0, l1b=0, l1c=0, l2=0, l1_custom=None, l1b_custom=None, weights=None, weights_neg=None):
    update = utilities.average_product(X1_p, X2_p, c1=n_c1, c2=n_c2, mean1=mean1, mean2=mean2, weights=weights) - \
        utilities.average_product(
            X1_n, X2_n, c1=n_c1, c2=n_c2, mean1=mean1, mean2=mean2, weights=weights_neg)
    if l2 > 0:
        update -= l2 * W
    if l1 > 0:
        update -= l1 * np.sign(W)
    if l1b > 0:  # NOT SUPPORTED FOR POTTS
        if n_c2 > 1:  # Potts RBM.
            update -= l1b * \
                np.sign(W) * \
                np.abs(W).mean(-1).mean(-1)[:, np.newaxis, np.newaxis]
        else:
            update -= l1b * np.sign(W) * (np.abs(W).sum(1))[:, np.newaxis]
    if l1c > 0:  # NOT SUPPORTED FOR POTTS
        update -= l1c * np.sign(W) * ((np.abs(W).sum(1))**2)[:, np.newaxis]
    if l1_custom is not None:
        update -= l1_custom * np.sign(W)
    if l1b_custom is not None:
        update -= l1b_custom[0] * (l1b_custom[1] * np.abs(W)).mean(-1).mean(-1)[
            :, np.newaxis, np.newaxis] * np.sign(W)

    if weights is not None:
        update *= weights.sum() / X1_p.shape[0]
    return update


def couplings_gradients_h(W, X1_p, X1_n, X2_p, X2_n, n_c1, n_c2, l1=0, l1b=0, l1c=0, l2=0, l1_custom=None, l1b_custom=None, weights=None, weights_neg=None):
    update = utilities.average_product(X1_p, X2_p, c1=n_c1, c2=n_c2, mean1=True, mean2=False, weights=weights) - \
        utilities.average_product(
            X1_n, X2_n, c1=n_c1, c2=n_c2, mean1=False, mean2=True, weights=weights_neg)
    if l2 > 0:
        update -= l2 * W
    if l1 > 0:
        update -= l1 * np.sign(W)
    if l1b > 0:  # NOT SUPPORTED FOR POTTS
        if n_c2 > 1:  # Potts RBM.
            update -= l1b * \
                np.sign(W) * \
                np.abs(W).mean(-1).mean(-1)[:, np.newaxis, np.newaxis]
        else:
            update -= l1b * np.sign(W) * (np.abs(W).sum(1))[:, np.newaxis]
    if l1c > 0:  # NOT SUPPORTED FOR POTTS
        update -= l1c * np.sign(W) * ((np.abs(W).sum(1))**2)[:, np.newaxis]
    if l1_custom is not None:
        update -= l1_custom * np.sign(W)
    if l1b_custom is not None:
        update -= l1b_custom[0] * (l1b_custom[1] * np.abs(W)).mean(-1).mean(-1)[
            :, np.newaxis, np.newaxis] * np.sign(W)

    if weights is not None:
        update *= weights.sum() / X1_p.shape[0]
    return update


def gauge_adjust_couplings(W, n_c1, n_c2, gauge='zerosum'):
    if gauge == 'zerosum':
        if (n_c1 > 1) & (n_c2 > 1):
            W = W  # To be changed...
        elif (n_c1 == 1) & (n_c2 > 1):
            W -= W.sum(-1)[:, :, np.newaxis] / n_c2
        elif (n_c1 > 1) & (n_c2 == 1):
            W -= W.sum(-1)[:, :, np.newaxis] / n_c1
    else:
        print('adjust_couplings -> gauge not supported')
    return W


class PGM(object):
    def __init__(self, n_layers=3, layers_name=['layer1', 'layer2', 'layer3'], layers_size=[100, 20, 30], layers_nature=['Bernoulli', 'Bernoulli', 'Bernoulli'], layers_n_c=[None, None, None]):
        self.n_layers = n_layers
        self.layers_name = layers_name
        self.layers_size = layers_size
        self.layers_nature = layers_nature
        self.layers_n_c = layers_n_c

    def markov_step(self, config, beta=1):
        return config

    def markov_step_PT(self, config):
        for i, beta in zip(np.arange(self.N_PT), self.betas):
            config[i] = self.markov_step(config[i], beta=beta)
        return config

    def markov_step_and_energy(self, config, E, beta=1):
        return config, E

    def exchange_step(self, x, permutation, F, F_swapped):
        if self.record_swaps:
            particle_id = self.particle_id[-1].copy()
        for i in utilities.get_indices_swaps(self.N_PT, self.count_swaps):
            log_proba = np.minimum(
                0, F[i] + F[i + 1] - F_swapped[i] - F_swapped[i + 1])
            swap = self.random_state.rand(*log_proba.shape) < np.exp(log_proba)
            for x_ in x:
                x_[i:i + 2][:, swap] = x_[permutation[i:i + 2]][:, swap]
            if self.record_swaps:
                particle_id[i:i + 2][:,
                                     swap] = particle_id[permutation[i:i + 2]][:, swap]
            if self.record_acceptance:
                self.acceptance_rates[i] = np.exp(log_proba).mean()
                self.log_acceptance_rates[i] = np.exp(log_proba.mean())
                self.mav_acceptance_rates[i] = self.mavar_gamma * \
                    self.mav_acceptance_rates[i] + \
                    self.acceptance_rates[i] * (1 - self.mavar_gamma)
                self.mav_log_acceptance_rates[i] = np.exp(self.mavar_gamma * np.log(
                    self.mav_log_acceptance_rates[i]) + np.log(self.log_acceptance_rates[i]) * (1 - self.mavar_gamma))

        if self.record_swaps:
            self.particle_id.append(particle_id)
        self.count_swaps += 1
        return x

    def gen_data(self, Nchains=10, Lchains=100, Nthermalize=0, Nstep=1, N_PT=1, config_init=[], beta=1, batches=None, reshape=True, record_replica=False,
                 record_acceptance=True, update_betas=False, update_betas_lr=0.025, update_betas_lr_decay=0.99, record_swaps=False,
                 MoI=False, MoI_h=False, PTv=False, PTh=False, reset_MoI=False):
        """
        Generate Monte Carlo samples from the RBM. Starting from random initial conditions, Gibbs updates are performed to sample from equilibrium.
        Inputs :
            Nchains (10): Number of Markov chains
            Lchains (100): Length of each chain
            Nthermalize (0): Number of Gibbs sampling steps to perform before the first sample of a chain.
            Nstep (1): Number of Gibbs sampling steps between each sample of a chain
            N_PT (1): Number of Monte Carlo Exchange replicas to use. This==useful if the mixing rate==slow. Watch self.acceptance_rates_g to check that it==useful (acceptance rates about 0==useless)
            batches (10): Number of batches. Must divide Nchains. higher==better for speed (more vectorized) but more RAM consuming.
            reshape (True): If True, the output==(Nchains x Lchains, n_visibles/ n_hiddens) (chains mixed). Else, the output==(Nchains, Lchains, n_visibles/ n_hiddens)
            config_init ([]). If not [], a Nchains X n_visibles numpy array that specifies initial conditions for the Markov chains.
            beta (1): The inverse temperature of the model.
        """
        if batches == None:
            batches = Nchains
        n_iter = Nchains // batches
        Ndata = Lchains * batches
        if record_replica:
            reshape = False

        if (N_PT > 1):
            if update_betas:
                record_acceptance = True
                self.update_betas_lr = update_betas_lr
                self.update_betas_lr_decay = update_betas_lr_decay
            if record_acceptance:
                self.mavar_gamma = 0.95

            if bool(MoI) | bool(MoI_h):
                self.from_MoI = bool(MoI)
                self.from_MoI_h = bool(MoI_h)
                # if interpolate_z:
                #     self.update_interpolation_MoI_lr = update_MoI_lr
                #     self.update_interpolation_MoI_lr_decay = update_MoI_lr_decay
                #     if degree_interpolate_z is None:
                #         degree_interpolate_z = N_PT-1
                # if reset_MoI or not hasattr(self,'zlayer'):
                #     if self.from_MoI:
                #         self.init_zlayer_APT(MoI,interpolate=interpolate_z,degree_interpolate=degree_interpolate_z,layer_id=0)
                #     else:
                #         self.init_zlayer_APT(MoI_h,interpolate=interpolate_z,degree_interpolate=degree_interpolate_z,layer_id=1)
                self.n_layers += 1  # Add extra layer.
                self.layers_name.append('zlayer')
                self.layers_size.append(1)
                self.layers_nature.append('Potts')
                self.layers_n_c.append(self.zlayer.n_c)
            else:
                self.from_MoI = False
                self.from_MoI_h = False

            self.PTv = PTv
            self.PTh = PTh
        else:
            record_acceptance = False
            update_betas = False
            self.from_MoI = False
            self.from_MoI_h = False
            self.PTv = False
            self.PTh = False
        self.record_acceptance = record_acceptance

        if (N_PT > 1) & record_replica:
            data = [np.zeros([Nchains, N_PT, Lchains, self.layers_size[i]], dtype=getattr(
                self, self.layers_name[i]).type) for i in range(self.n_layers)]
        else:
            data = [np.zeros([Nchains, Lchains, self.layers_size[i]], dtype=getattr(
                self, self.layers_name[i]).type) for i in range(self.n_layers)]

        if self.n_layers == 1:
            data = data[0]

        if config_init != []:
            if type(config_init) == np.ndarray:
                config_init_ = []
                for k, layer_ in enumerate(self.layers_name):
                    if config_init.shape[1] == self.layers_size[k]:
                        config_init_.append(config_init.copy())
                    else:
                        config_init_.append(
                            getattr(self, layer_).random_init_config(batches))
                config_init = config_init_

        for i in range(n_iter):
            if config_init == []:
                config = self._gen_data(Nthermalize, Ndata, Nstep, N_PT=N_PT, batches=batches, reshape=False,
                                        beta=beta, record_replica=record_replica, update_betas=update_betas, record_swaps=record_swaps)
            else:
                config = self._gen_data(Nthermalize, Ndata, Nstep, N_PT=N_PT, batches=batches, reshape=False, beta=beta, record_replica=record_replica, config_init=[
                                        config_init[l][batches * i:batches * (i + 1)] for l in range(self.n_layers)], update_betas=update_betas, record_swaps=record_swaps)
            if (N_PT > 1) & record_replica:
                if self.n_layers == 1:
                    data[batches * i:batches *
                         (i + 1), :, :, :] = copy.copy(np.swapaxes(config, 0, 2))
                else:
                    for l in range(self.n_layers):
                        data[l][batches * i:batches *
                                (i + 1), :, :, :] = copy.copy(np.swapaxes(config[l], 0, 2))
            else:
                if self.n_layers == 1:
                    data[batches * i:batches *
                         (i + 1), :, :] = copy.copy(np.swapaxes(config, 0, 1))
                else:
                    for l in range(self.n_layers):
                        data[l][batches * i:batches *
                                (i + 1), :, :] = copy.copy(np.swapaxes(config[l], 0, 1))

        if reshape:
            if self.n_layers == 1:
                data = data.reshape([Nchains * Lchains, self.layers_size[0]])
            else:
                data = [data[layer].reshape(
                    [Nchains * Lchains, self.layers_size[layer]]) for layer in range(self.n_layers)]

        if self.from_MoI | self.from_MoI_h:  # Remove extra layer.
            self.n_layers -= 1
            self.layers_name = self.layers_name[:-1]
            self.layers_size = self.layers_size[:-1]
            self.layers_nature = self.layers_nature[:-1]
            self.layers_n_c = self.layers_n_c[:-1]
        return data

    def _gen_data(self, Nthermalize, Ndata, Nstep, N_PT=1, batches=1, reshape=True, config_init=[], beta=1, record_replica=False, update_betas=False, record_swaps=False):
        self.N_PT = N_PT
        if self.N_PT > 1:
            if update_betas | (not hasattr(self, 'betas')):
                self.betas = np.arange(N_PT) / (N_PT - 1) * beta
                self.betas = self.betas[::-1].astype(curr_float)
            if (len(self.betas) != N_PT):
                self.betas = np.arange(N_PT) / (N_PT - 1) * beta
                self.betas = self.betas[::-1].astype(curr_float)

            self.acceptance_rates = np.zeros(N_PT - 1, dtype=curr_float)
            self.mav_acceptance_rates = np.zeros(N_PT - 1, dtype=curr_float)
            self.log_acceptance_rates = np.ones(N_PT - 1, dtype=curr_float)
            self.mav_log_acceptance_rates = np.ones(N_PT - 1, dtype=curr_float)
        self.count_swaps = 0
        self.record_swaps = record_swaps
        if self.record_swaps:
            self.particle_id = [
                np.arange(N_PT)[:, np.newaxis].repeat(batches, axis=1)]

        Ndata = Ndata // batches
        if N_PT > 1:
            config = [getattr(self, layer).random_init_config(
                batches, N_PT=N_PT) for layer in self.layers_name]
            if config_init != []:
                for l in range(self.n_layers):
                    config[l][0] = config_init[l]

            if not (self.from_MoI | self.from_MoI_h):
                energy = self.energy(config, remove_init=True)
        else:
            if config_init != []:
                config = config_init
            else:
                config = [getattr(self, layer).random_init_config(
                    batches) for layer in self.layers_name]

        if self.n_layers == 1:
            config = config[0]  # no list

        for _ in range(Nthermalize):
            if N_PT > 1:
                if self.from_MoI:
                    config = self.exchange_step_APT(config)
                    config = self.markov_step_APT(
                        config, beta=self.betas, recompute=False)
                elif self.from_MoI_h:
                    config = self.exchange_step_APTh(config)
                    config = self.markov_step_APTh(
                        config, beta=self.betas, recompute=False)
                elif self.PTv:
                    config = self.exchange_step_PTv(config)
                    config = self.markov_step(
                        config, beta=self.betas, recompute=False)
                elif self.PTh:
                    config = self.exchange_step_PTh(config)
                    config = self.markov_step_h(
                        config, beta=self.betas, recompute=False)
                else:
                    config, energy = self.exchange_step_PT(config, energy)
                    config, energy = self.markov_step_and_energy(
                        config, energy, beta=self.betas)
                if update_betas:
                    self.update_betas(beta=beta)
                    self.update_betas_lr *= self.update_betas_lr_decay
            else:
                config = self.markov_step(config, beta=beta)
        if self.n_layers == 1:
            data = [utilities.copy_config(
                config, N_PT=N_PT, record_replica=record_replica)]
        else:
            data = [[utilities.copy_config(
                config[l], N_PT=N_PT, record_replica=record_replica)] for l in range(self.n_layers)]

        for _ in range(Ndata - 1):
            for _ in range(Nstep):
                if N_PT > 1:
                    if self.from_MoI:
                        config = self.exchange_step_APT(config)
                        config = self.markov_step_APT(
                            config, beta=self.betas, recompute=False)
                    elif self.from_MoI_h:
                        config = self.exchange_step_APTh(config)
                        config = self.markov_step_APTh(
                            config, beta=self.betas, recompute=False)
                    elif self.PTv:
                        config = self.exchange_step_PTv(config)
                        config = self.markov_step(
                            config, beta=self.betas, recompute=False)
                    elif self.PTh:
                        config = self.exchange_step_PTh(config)
                        config = self.markov_step_h(
                            config, beta=self.betas, recompute=False)
                    else:
                        config, energy = self.exchange_step_PT(config, energy)
                        config, energy = self.markov_step_and_energy(
                            config, energy, beta=self.betas)

                    if update_betas:
                        self.update_betas(beta=beta)
                        self.update_betas_lr *= self.update_betas_lr_decay
                else:
                    config = self.markov_step(config, beta=beta)
            if self.n_layers == 1:
                data.append(utilities.copy_config(
                    config, N_PT=N_PT, record_replica=record_replica))
            else:
                for l in range(self.n_layers):
                    data[l].append(utilities.copy_config(
                        config[l], N_PT=N_PT, record_replica=record_replica))

        if self.record_swaps:
            print('cleaning particle trajectories')
            positions = np.array(self.particle_id)
            invert = np.zeros([batches, Ndata + Nthermalize, N_PT])
            for b in range(batches):
                for i in range(Ndata + Nthermalize):
                    for k in range(N_PT):
                        invert[b, i, k] = np.nonzero(
                            positions[i, :, b] == k)[0]
            self.particle_id = invert
            self.last_at_zero = np.zeros([batches, Ndata + Nthermalize, N_PT])
            for b in range(batches):
                for i in range(Ndata + Nthermalize):
                    for k in range(N_PT):
                        tmp = np.nonzero(self.particle_id[b, :i, k] == 0)[0]
                        if len(tmp) > 0:
                            self.last_at_zero[b, i, k] = i - 1 - tmp.max()
                        else:
                            self.last_at_zero[b, i, k] = i
            self.last_at_zero[:, 0, 0] = 0

            self.trip_duration = np.zeros([batches, Ndata + Nthermalize])
            for b in range(batches):
                for i in range(Ndata + Nthermalize):
                    self.trip_duration[b, i] = self.last_at_zero[b, i, np.nonzero(
                        invert[b, i, :] == N_PT - 1)[0]]

        if reshape:
            if self.n_layers == 1:
                data = np.array(data).reshape(
                    [Ndata * batches, self.layers_size[0]])
            else:
                for l in range(self.n_layers):
                    data[l] = np.array(data[l]).reshape(
                        [Ndata * batches, self.layers_size[l]])
        else:
            if self.n_layers == 1:
                data = np.array(data)
            else:
                for l in range(self.n_layers):
                    data[l] = np.array(data[l])
        return data

    def update_betas(self, beta=1):
        if self.N_PT < 3:
            return
        else:
            if self.acceptance_rates.mean() > 0:
                # self.stiffness = np.maximum(1 - (self.mav_log_acceptance_rates/self.mav_log_acceptance_rates.mean()),0) + 1e-4 * np.ones(self.N_PT-1)
                self.stiffness = np.maximum(
                    1 - (self.mav_acceptance_rates / self.mav_acceptance_rates.mean()), 0) + 1e-4 * np.ones(self.N_PT - 1)
                diag = self.stiffness[0:-1] + self.stiffness[1:]
                if self.N_PT > 3:
                    offdiag_g = -self.stiffness[1:-1]
                    offdiag_d = -self.stiffness[1:-1]
                    M = diags([offdiag_g, diag, offdiag_d], offsets=[-1, 0, 1],
                              shape=[self.N_PT - 2, self.N_PT - 2]).toarray()
                else:
                    M = diags([diag], offsets=[0], shape=[
                              self.N_PT - 2, self.N_PT - 2]).toarray()
                B = np.zeros(self.N_PT - 2, dtype=curr_float)
                B[0] = self.stiffness[0] * beta
                self.betas[1:-1] = self.betas[1:-1] * \
                    (1 - self.update_betas_lr) + \
                    self.update_betas_lr * np.linalg.solve(M, B)
                # self.update_betas_lr*= self.update_betas_lr_decay

    def AIS(self, M=10, n_betas=10000, batches=None, verbose=0, beta_type='linear', reset_betas=True):
        # AIS for non-linear interpolation currently not supported. Copy first the model..
        if self.interpolate:
            if verbose:
                print('Interpolate=true, Making a copy of the model first...')
            model = copy.deepcopy(self)
            for layer_ in self.layers_name:
                L = getattr(model, layer_)
                if L.position == 'visible':
                    target = self.moments_data
                else:
                    target = None
                L.init_params_from_data(target, value='moments')
                for param in L.list_params:
                    L.__dict__[param] = getattr(self, layer_).__dict__[param]
                    L.__dict__[param + '1'] *= 0
        else:
            model = self

        if beta_type == 'linear':
            betas = np.arange(n_betas) / float(n_betas - 1)
        elif beta_type == 'root':
            betas = np.sqrt(np.arange(n_betas) / float(n_betas - 1))
        elif beta_type == 'adaptive':
            if not hasattr(model, 'N_PT'):
                model.N_PT = 1
            copy_N_PT = copy.copy(model.N_PT)
            if hasattr(model, 'betas'):
                if reset_betas:
                    tmp = 0
                else:
                    if model.N_PT % 2:
                        tmp = 1
                    else:
                        tmp = 0
            else:
                tmp = 0

            if tmp:
                if verbose:
                    print('Using previously computed betas: %s' % model.betas)
                N_PT = len(model.betas)
                tmp2 = 0
            else:
                if hasattr(model, 'betas'):
                    tmp2 = 1
                    copy_beta = model.betas.copy()
                else:
                    tmp2 = 0

                Nthermalize = 200
                Nchains = 20
                N_PT = 11
                model.adaptive_PT_lr = 0.05
                model.adaptive_PT_decay = True
                model.adaptive_PT_lr_decay = 10**(-1 / float(Nthermalize))
                if verbose:
                    t = time.time()
                    print('Learning betas...')
                model.gen_data(N_PT=N_PT, Nchains=Nchains, Lchains=1,
                               Nthermalize=Nthermalize, update_betas=True)
                if verbose:
                    print('Elapsed time: %s, Acceptance rates: %s' %
                          (time.time() - t, model.mav_acceptance_rates))
            betas = []
            sparse_betas = model.betas[::-1]
            for i in range(N_PT - 1):
                betas += list(sparse_betas[i] + (sparse_betas[i + 1] - sparse_betas[i]) * np.arange(
                    n_betas / (N_PT - 1)) / float(n_betas / (N_PT - 1) - 1))
            betas = np.array(betas)
            n_betas = len(betas)
            # if verbose:
            # import matplotlib.pyplot as plt
            # plt.plot(betas); plt.title('Interpolating temperatures');plt.show()

            # Initialization.
        log_weights = np.zeros(M, dtype=curr_float)
        config = []
        layers_name = model.layers_name
        layers_size = model.layers_size
        layers_n_c = model.layers_n_c
        n_layers = model.n_layers

        for layer_, N, n_c in zip(layers_name, layers_size, layers_n_c):
            if n_c > 1:
                config.append(getattr(model, layer_).sample_from_inputs(
                    np.zeros([M, N, n_c], dtype=curr_float), beta=0))
            else:
                config.append(getattr(model, layer_).sample_from_inputs(
                    np.zeros([M, N], dtype=curr_float), beta=0))

        if n_layers == 1:
            fields_eff = model.compute_fields_eff(config[0][0])
            config = (config[0][0], fields_eff)
        energy = np.zeros(M, dtype=curr_float)

        log_Z_init = np.zeros(1, dtype=curr_float)
        for N, layer_ in zip(layers_size, layers_name):
            log_Z_init += getattr(model, layer_).logpartition(None, beta=0)

        if verbose:
            print('Initial evaluation: log(Z) = %s' % log_Z_init)

        for i in range(1, n_betas):
            if verbose:
                if (i % 2000 == 0):
                    print('Iteration %s, beta: %s' % (i, betas[i]))
                    print('Current evaluation: log(Z)= %s +- %s' % ((log_Z_init +
                                                                     log_weights).mean(), (log_Z_init + log_weights).std() / np.sqrt(M)))

            config, energy = model.markov_step_and_energy(
                config, energy, beta=betas[i])
            log_weights += -(betas[i] - betas[i - 1]) * energy
        if beta_type == 'adaptive':
            model.N_PT = copy_N_PT
            if tmp2:
                model.betas = copy_beta

        self.log_Z_AIS = (log_Z_init + log_weights).mean()
        self.log_Z_AIS_std = (log_Z_init + log_weights).std() / np.sqrt(M)

        if verbose:
            print('Final evaluation: log(Z)= %s +- %s' %
                  (self.log_Z_AIS, self.log_Z_AIS_std))
        return self.log_Z_AIS, self.log_Z_AIS_std

    def likelihood(self, data, recompute_Z=False):
        if (not hasattr(self, 'log_Z_AIS')) | recompute_Z:
            self.AIS()
        return -self.free_energy(data) - self.log_Z_AIS

    def init_zlayer_APT(self, MoI, interpolate=False, degree_interpolate=2, layer_id=0):
        xlayer = getattr(self, self.layers_name[layer_id])
        n_c = xlayer.n_c
        MoI_exists = (type(MoI) != int)

        if MoI_exists:
            n_cz = MoI.M
        else:
            n_cz = MoI

        if interpolate:
            self.zlayer = layer.initLayer(
                N=1, nature='PottsInterpolate', n_c=n_cz, degree=degree_interpolate)
        else:
            self.zlayer = layer.initLayer(N=1, nature='Potts', n_c=n_cz)
        self.zlayer.mu = np.ones([1, n_cz], dtype=curr_float) / n_cz
        # self.zlayer.average_likelihood = np.zeros(n_cz,dtype=curr_float)

        if MoI_exists:
            self.zlayer.mu = MoI.muh[np.newaxis]
            tmp = MoI.weights - xlayer.fields0[np.newaxis]
            if n_c > 1:
                self.weights_MoI = np.asarray(np.swapaxes(
                    tmp[np.newaxis, :, :, :], 1, 2), order='c').astype(curr_float)
            else:
                self.weights_MoI = np.asarray(np.swapaxes(
                    tmp[np.newaxis, :, :], 1, 2), order='c').astype(curr_float)
        else:
            if n_c > 1:
                self.weights_MoI = np.sqrt(
                    0.01 / xlayer.N) * np.random.randn(1, xlayer.N, n_cz, n_c).astype(curr_float)
            else:
                self.weights_MoI = np.sqrt(
                    0.01 / xlayer.N) * np.random.randn(1, xlayer.N, n_cz).astype(curr_float)

        self.update_params_MoI(layer_id=layer_id)
        return

    def update_params_MoI(self, layer_id=0, eps=1e-4, verbose=True):
        xlayer = getattr(self, self.layers_name[layer_id])
        zlayer = self.zlayer
        k = np.argmin(zlayer.mu)
        if zlayer.mu[0, k] < 0.02 / zlayer.n_c:
            if verbose:
                print('Reloading mixture %s, %.2e' % (k, zlayer.mu[0, k]))
            # kmax = np.argmin(zlayer.average_likelihood * (zlayer.mu[0]>0.5/zlayer.n_c ))
            kmax = np.argmax(zlayer.mu[0])
            zlayer.mu[0, kmax] = (zlayer.mu[0, kmax] + zlayer.mu[0, k]) / 2
            zlayer.mu[0, k] = zlayer.mu[0, kmax].copy()
            noise = 1e-3 * \
                np.random.randn(
                    *self.weights_MoI[0, :, 0].shape).astype(curr_float)
            self.weights_MoI[:, :, k] = self.weights_MoI[:, :, kmax] + noise
            self.weights_MoI[:, :, kmax] -= noise
            if hasattr(self, 'fantasy_z'):
                self.fantasy_z[(self.fantasy_z == k) | (self.fantasy_z == kmax)] = k + (kmax - k) * self.random_state.randint(
                    0, high=2, size=self.fantasy_z[(self.fantasy_z == k) | (self.fantasy_z == kmax)].shape)

            if self.interpolate_z:
                zlayer.fields1[:, :, k] = zlayer.fields1[:, :, kmax]
                zlayer.fields1 -= zlayer.fields1.mean(-1)[..., np.newaxis]

            for attr in ['gradient_moment1', 'gradient_moment2', 'previous_update']:
                if hasattr(self, attr):
                    self.__dict__[attr]['weights_MoI'][:, :, k] = self.__dict__[
                        attr]['weights_MoI'][:, :, kmax].copy()
                    if self.interpolate_z:
                        self.__dict__[attr]['zlayer']['fields1'][:, :, k] = self.__dict__[
                            attr]['zlayer']['fields1'][:, :, kmax].copy()

        zlayer.fields = np.log(zlayer.mu + eps)
        zlayer.fields -= zlayer.fields.mean()
        zlayer.fields0 = zlayer.fields - \
            xlayer.logpartition(None, I0=np.swapaxes(
                self.weights_MoI[0], 0, 1), beta=0)[np.newaxis]
        zlayer.fields0 -= zlayer.fields0.mean()
        zlayer._target_moments0 = (zlayer.mu,)
        zlayer._mean_weight0 = 1.

        self.muzx = xlayer.mean_from_inputs(
            None, I0=np.swapaxes(self.weights_MoI[0], 0, 1), beta=0)
        if xlayer.n_c > 1:
            self.muzx *= self.zlayer.mu[0, :, np.newaxis, np.newaxis]
        else:
            self.muzx *= self.zlayer.mu[0, :, np.newaxis]
        self.muzx = np.swapaxes(self.muzx, 0, 1)[np.newaxis]
        return
