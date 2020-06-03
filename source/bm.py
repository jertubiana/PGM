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
import utilities
import numba_utilities as cy_utilities
import time
from float_precision import curr_float, curr_int


class BM(pgm.PGM):
    def __init__(self, N=100, nature='Bernoulli', n_c=1, random_state=None, gauge='zerosum', zero_field=False):
        self.N = N
        self.nature = nature
        self.random_state = utilities.check_random_state(random_state)
        if self.nature == 'Potts':
            self.n_c = n_c
        else:
            self.n_c = 1
        self.zero_field = zero_field

        super(BM, self).__init__(n_layers=1, layers_size=[self.N], layers_nature=[
            self.nature + '_coupled'], layers_n_c=[self.n_c], layers_name=['layer'])

        self.gauge = gauge

        self.layer = layer.initLayer(N=self.N, nature=self.nature + '_coupled', position='visible',
                                     n_c=self.n_c, random_state=self.random_state, zero_field=self.zero_field, gauge=self.gauge)
        self.init_couplings(0.01)
        self.tmp_l2_fields = 0

    def init_couplings(self, amplitude):
        if (self.n_c > 1):
            self.layer.couplings = (amplitude *
                                    self.random_state.randn(self.N, self.N, self.n_c, self.n_c)).astype(curr_float)
            self.layer.couplings = pgm.gauge_adjust_couplings(
                self.layer.couplings, self.n_c, self.n_c, gauge=self.gauge)
        else:
            self.layer.couplings = (amplitude *
                                    self.random_state.randn(self.N, self.N)).astype(curr_float)

    def markov_step_PT(self, config):
        x, fields_eff = config
        for i, beta in zip(np.arange(self.N_PT), self.betas):
            x[i], fields_eff[i] = self.markov_step(
                (x[i], fields_eff[i]), beta=beta)
        return (x, fields_eff)

    def markov_step_PT2(self, config, E):
        x, fields_eff = config
        for i, beta in zip(np.arange(self.N_PT), self.betas):
            (x[i], fields_eff[i]), E[i] = self.layer.sample_and_energy_from_inputs(
                None, beta=beta, previous=(x[i], fields_eff[i]), remove_init=True)
        return (x, fields_eff), E

    def exchange_step_PT(self, config, E, record_acceptance=True, compute_energy=True):
        x, fields_eff = config
        if compute_energy:
            for i in np.arange(self.N_PT):
                E[i, :] = self.energy(x[i, :, :], remove_init=True)

        if self.record_swaps:
            particle_id = self.particle_id[-1].copy()
        for i in np.arange(self.count_swaps % 2, self.N_PT - 1, 2):
            proba = np.minimum(1,  np.exp(
                (self.betas[i + 1] - self.betas[i]) * (E[i + 1, :] - E[i, :])))
            swap = np.random.rand(proba.shape[0]) < proba
            if i > 0:
                x[i:i + 2, swap] = x[i + 1:i - 1:-1, swap]
                fields_eff[i:i + 2, swap] = fields_eff[i + 1:i - 1:-1, swap]
                E[i:i + 2, swap] = E[i + 1:i - 1:-1, swap]
                if self.record_swaps:
                    particle_id[i:i + 2,
                                swap] = particle_id[i + 1:i - 1:-1, swap]

            else:
                x[i:i + 2, swap] = x[i + 1::-1, swap]
                fields_eff[i:i + 2, swap] = fields_eff[i + 1::-1, swap]
                E[i:i + 2, swap] = E[i + 1::-1, swap]
                if self.record_swaps:
                    particle_id[i:i + 2, swap] = particle_id[i + 1::-1, swap]

            if record_acceptance:
                self.acceptance_rates[i] = swap.mean()
                self.mav_acceptance_rates[i] = self.mavar_gamma * \
                    self.mav_acceptance_rates[i] + \
                    self.acceptance_rates[i] * (1 - self.mavar_gamma)

        if self.record_swaps:
            self.particle_id.append(particle_id)

        self.count_swaps += 1
        return (x, fields_eff), E

    def update_betas(self, beta=1):
        super(BM, self).update_betas(beta=beta)

    def markov_step(self, config, beta=1):
        x, fields_eff = config
        (x, fields_eff) = self.layer.sample_from_inputs(
            None, previous=(x, fields_eff), beta=beta)
        return (x, fields_eff)

    def markov_step_and_energy(self, config, E, beta=1):
        x, fields_eff = config
        (x, fields_eff), E = self.layer.sample_and_energy_from_inputs(
            None, beta=beta, previous=(x, fields_eff), remove_init=True)
        return (x, fields_eff), E

    def compute_fields_eff(self, x, beta=1):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return self.layer.compute_output(x, self.layer.couplings) + self.layer.fields[np.newaxis]

    def energy(self, x, remove_init=False):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return self.layer.energy(x, remove_init=remove_init)

    def free_energy(self, x):
        return self.energy(x)

    def compute_all_moments(self):
        configurations = utilities.make_all_discrete_configs(
            self.N, self.nature, c=self.n_c)
        weights = -self.free_energy(configurations)
        maxi = weights.max()
        weights -= maxi
        weights = np.exp(weights)
        Z = weights.sum()
        if self.nature == 'Potts':
            mean = np.zeros([self.N, self.n_c])
            for color in range(self.n_c):
                mean[:, color] = ((configurations == color)
                                  * weights[:, np.newaxis]).sum(0) / Z
            covariance = np.zeros([self.N, self.N, self.n_c, self.n_c])
            for color1 in range(self.n_c):
                for color2 in range(self.n_c):
                    covariance[:, :, color1, color2] = ((configurations[:, np.newaxis, :] == color2) * (
                        configurations[:, :, np.newaxis] == color1) * weights[:, np.newaxis, np.newaxis]).sum(0) / Z
        else:
            mean = (configurations * weights[:, np.newaxis]).sum(0) / Z
            covariance = (configurations[:, np.newaxis, :] * configurations[:,
                                                                            :, np.newaxis] * weights[:, np.newaxis, np.newaxis]).sum(0) / Z

        Z = Z * np.exp(maxi)
        return Z, mean, covariance

    def pseudo_likelihood(self, x):
        if self.nature not in ['Bernoulli', 'Spin', 'Potts']:
            print('PL not supported for continuous data')
        else:
            fields = self.compute_fields_eff(x)
            if self.nature == 'Bernoulli':
                return (fields * x - np.logaddexp(fields, 0)).mean(1)
            elif self.nature == 'Spin':
                return (fields * x - np.logaddexp(fields, -fields)).mean(1)
            elif self.nature == 'Potts':
                return (cy_utilities.substitute_C(fields, x) - utilities.logsumexp(fields, axis=2)).mean(1)

    def gen_data(self, Nchains=10, Lchains=100, Nthermalize=0, Nstep=1, N_PT=1, config_init=[], beta=1, batches=None, reshape=True, record_replica=False, record_acceptance=None, update_betas=None, record_swaps=False):
        return super(BM, self).gen_data(Nchains=Nchains, Lchains=Lchains, Nthermalize=Nthermalize, Nstep=Nstep, N_PT=N_PT, config_init=config_init, beta=beta, batches=batches, reshape=reshape, record_replica=record_replica, record_acceptance=record_acceptance, update_betas=update_betas, record_swaps=record_swaps)

    def fit(self, data, batch_size=100, nchains=100, learning_rate=None, extra_params=None, init='independent', optimizer='SGD', N_PT=1, N_MC=1, n_iter=10,
            lr_decay=True, lr_final=None, decay_after=0.5, l1=0, l1b=0, l1c=0, l2=0, l2_fields=0, no_fields=False, batch_norm=False,
            update_betas=None, record_acceptance=None, epsilon=1e-6, verbose=1, record=[], record_interval=100, p=[1, 0, 0], pseudo_count=0, weights=None):

        self.nchains = nchains
        self.optimizer = optimizer
        self.record_swaps = False
        self.batch_norm = batch_norm
        self.layer.batch_norm = batch_norm

        self.n_iter = n_iter

        if learning_rate is None:
            if self.nature in ['Bernoulli', 'Spin', 'Potts']:
                learning_rate = 0.1
            else:
                learning_rate = 0.01

            if self.optimizer == 'ADAM':
                learning_rate *= 0.1

        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        if self.lr_decay:
            self.decay_after = decay_after
            self.start_decay = self.n_iter * self.decay_after
            if lr_final is None:
                self.lr_final = 1e-2 * self.learning_rate
            else:
                self.lr_final = lr_final
            self.decay_gamma = (float(self.lr_final) / float(self.learning_rate)
                                )**(1 / float(self.n_iter * (1 - self.decay_after)))

        self.gradient = self.initialize_gradient_dictionary()

        if self.optimizer == 'momentum':
            if extra_params is None:
                extra_params = 0.9
            self.momentum = extra_params
            self.previous_update = self.initialize_gradient_dictionary()

        elif self.optimizer == 'ADAM':
            if extra_params is None:
                extra_params = [0.9, 0.999, 1e-8]
            self.beta1 = extra_params[0]
            self.beta2 = extra_params[1]
            self.epsilon = extra_params[2]

            self.gradient_moment1 = self.initialize_gradient_dictionary()
            self.gradient_moment2 = self.initialize_gradient_dictionary()

        if weights is not None:
            weights = np.asarray(weights, dtype=float)

        mean = utilities.average(data, c=self.n_c, weights=weights)
        covariance = utilities.average_product(
            data, data, c1=self.n_c, c2=self.n_c, weights=weights)
        if pseudo_count > 0:
            p = data.shape[0] / float(data.shape[0] + pseudo_count)
            covariance = p**2 * covariance + p * \
                (1 - p) * (mean[np.newaxis, :, np.newaxis, :] * mean[:,
                                                                     np.newaxis, :, np.newaxis]) / self.n_c + (1 - p)**2 / self.n_c**2
            mean = p * mean + (1 - p) / self.n_c

        iter_per_epoch = data.shape[0] // batch_size
        if init != 'previous':
            norm_init = 0
            self.init_couplings(norm_init)
            if init == 'independent':
                self.layer.init_params_from_data(data, eps=epsilon, value='data')

        self.N_PT = N_PT
        self.N_MC = N_MC

        self.l1 = l1
        self.l1b = l1b
        self.l1c = l1c
        self.l2 = l2
        self.tmp_l2_fields = l2_fields
        self.no_fields = no_fields

        if self.N_PT > 1:
            if record_acceptance == None:
                record_acceptance = True
            self.record_acceptance = record_acceptance

            if update_betas == None:
                update_betas = True

            self._update_betas = update_betas

            if self.record_acceptance:
                self.mavar_gamma = 0.95
                self.acceptance_rates = np.zeros(N_PT - 1)
                self.mav_acceptance_rates = np.zeros(N_PT - 1)
            self.count_swaps = 0

            if self._update_betas:
                record_acceptance = True
                self.update_betas_lr = 0.1
                self.update_betas_lr_decay = 1

            if self._update_betas | (not hasattr(self, 'betas')):
                self.betas = np.arange(N_PT) / float(N_PT - 1)
                self.betas = self.betas[::-1]
            if (len(self.betas) != N_PT):
                self.betas = np.arange(N_PT) / float(N_PT - 1)
                self.betas = self.betas[::-1]

        if self.nature == 'Potts':
            (self.fantasy_x, self.fantasy_fields_eff) = self.layer.sample_from_inputs(
                np.zeros([self.N_PT * self.nchains, self.N, self.n_c]), beta=0)
        else:
            (self.fantasy_x, self.fantasy_fields_eff) = self.layer.sample_from_inputs(
                np.zeros([self.N_PT * self.nchains, self.N]), beta=0)
        if self.N_PT > 1:
            self.fantasy_x = self.fantasy_x.reshape(
                [self.N_PT, self.nchains, self.N])
            if self.nature == 'Potts':
                self.fantasy_fields_eff = self.fantasy_fields_eff.reshape(
                    [self.N_PT, self.nchains, self.N, self.n_c])
            else:
                self.fantasy_fields_eff = self.fantasy_fields_eff.reshape(
                    [self.N_PT, self.nchains, self.N])
            self.fantasy_E = np.zeros([self.N_PT, self.nchains])

        self.count_updates = 0
        if verbose:
            if weights is not None:
                lik = (self.pseudo_likelihood(data) *
                       weights).sum() / weights.sum()
            else:
                lik = self.pseudo_likelihood(data).mean()
            print('Iteration number 0, pseudo-likelihood: %.2f' % lik)

        result = {}
        if 'J' in record:
            result['J'] = []
        if 'F' in record:
            result['F'] = []

        count = 0

        for epoch in range(1, n_iter + 1):
            if verbose:
                begin = time.time()
            if self.lr_decay:
                if (epoch > self.start_decay):
                    self.learning_rate *= self.decay_gamma

            print('Starting epoch %s' % (epoch))
            for _ in range(iter_per_epoch):
                self.minibatch_fit(mean, covariance)

                if (count % record_interval == 0):
                    if 'J' in record:
                        result['J'].append(self.layer.couplings.copy())
                    if 'F' in record:
                        result['F'].append(self.layer.fields.copy())

                count += 1

            if verbose:
                end = time.time()
                if weights is not None:
                    lik = (self.pseudo_likelihood(data) *
                           weights).sum() / weights.sum()
                else:
                    lik = self.pseudo_likelihood(data).mean()

                print("[%s] Iteration %d, pseudo-likelihood = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, epoch,
                         lik, end - begin))

        return result

    def minibatch_fit(self, mean, covariance):
        self.count_updates += 1
        for _ in range(self.N_MC):
            if self.N_PT > 1:
                (self.fantasy_x, self.fantasy_fields_eff), self.fantasy_E = self.markov_step_PT2(
                    (self.fantasy_x, self.fantasy_fields_eff), self.fantasy_E)
                (self.fantasy_x, self.fantasy_fields_eff), self.fantasy_E = self.exchange_step_PT(
                    (self.fantasy_x, self.fantasy_fields_eff), self.fantasy_E, record_acceptance=self.record_acceptance, compute_energy=False)

#                (self.fantasy_x,self.fantasy_fields_eff) = self.markov_step_PT((self.fantasy_x,self.fantasy_fields_eff) )
#                (self.fantasy_x,self.fantasy_fields_eff),self.fantasy_E= self.exchange_step_PT((self.fantasy_x,self.fantasy_fields_eff),self.fantasy_E,record_acceptance=self.record_acceptance)
                if self._update_betas:
                    self.update_betas()

            else:
                (self.fantasy_x, self.fantasy_fields_eff) = self.markov_step(
                    (self.fantasy_x, self.fantasy_fields_eff))

        if self.N_PT > 1:
            X_neg = self.fantasy_x[0, :, :]
        else:
            X_neg = self.fantasy_x

        self.gradient['layer'] = self.layer.internal_gradients(
            (mean, covariance), X_neg, l1=self.l1, l2=self.l2, value='moments')
        if self.tmp_l2_fields > 0:
            self.gradient['layer']['fields'] -= self.tmp_l2_fields * \
                self.layer.fields

        for internal_param, gradient in self.gradient['layer'].items():
            current = getattr(self.layer, internal_param)
            if self.optimizer == 'SGD':
                new = current + self.learning_rate * gradient
            elif self.optimizer == 'momentum':
                self.previous_update['layer'][internal_param] = (
                    1 - self.momentum) * self.learning_rate * gradient + self.momentum * self.previous_update['layer'][internal_param]
                new = current + self.previous_update['layer'][internal_param]
            elif self.optimizer == 'ADAM':
                self.gradient_moment1['layer'][internal_param] = (
                    1 - self.beta1) * gradient + self.beta1 * self.gradient_moment1['layer'][internal_param]
                self.gradient_moment2['layer'][internal_param] = (
                    1 - self.beta2) * gradient**2 + self.beta2 * self.gradient_moment2['layer'][internal_param]

                new = current + self.learning_rate * (self.gradient_moment1['layer'][internal_param] / (1 - self.beta1**self.count_updates)) / (
                    self.epsilon + np.sqrt(self.gradient_moment2['layer'][internal_param] / (1 - self.beta2**self.count_updates)))

            setattr(self.layer, internal_param, new)

        if self.nature == 'Potts':
            self.layer.couplings = pgm.gauge_adjust_couplings(
                self.layer.couplings, self.n_c, self.n_c, gauge=self.gauge)
        self.layer.couplings[range(self.N), range(self.N)] *= 0

        if self.N_PT > 1:
            for i in range(self.N_PT):
                self.fantasy_fields_eff[i] = self.layer.fields[np.newaxis] + \
                    self.layer.compute_output(
                        self.fantasy_x[i], self.layer.couplings)
        else:
            self.fantasy_fields_eff = self.layer.fields[np.newaxis] + self.layer.compute_output(
                self.fantasy_x, self.layer.couplings)

        # if self.N_PT>1:
            # for i in range(self.N_PT):
            # self.fantasy_fields_eff[i] += self.learning_rate * (self.gradient['layer']['fields'][np.newaxis] + self.layer.compute_output(self.fantasy_x[i], self.gradient['layer']['couplings']))
        # else:
            # self.fantasy_fields_eff += self.learning_rate * (self.gradient['layer']['fields'][np.newaxis] + self.layer.compute_output(self.fantasy_x, self.gradient['layer']['couplings']))
            # print 'lololol'

    def initialize_gradient_dictionary(self):
        out = {}
        out['layer'] = self.layer.internal_gradients(
            np.zeros([1, self.N], dtype=int), np.zeros([1, self.N], dtype=int))
        return out
