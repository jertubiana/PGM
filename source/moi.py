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
import numba_utilities as cy_utilities
import bisect
import time
from float_precision import double_precision, curr_float, curr_int


def binarize(data, n_c):
    binary_data = np.zeros([data.shape[0], data.shape[1], n_c], dtype=curr_int)
    for c in range(n_c):
        binary_data[:, :, c] = (data == c)
    return binary_data


def tower_sampling(Ndata, cumulative):
    rng = np.random.rand(Ndata)
    L = []
    for i in range(Ndata):
        L.append(bisect.bisect_left(cumulative, rng[i]))
    return np.array(L)


def KMPP_choose_centroids(data, M, nmax=100000, verbose=False):
    if verbose:
        print('Start Choosing Centroids with KM++...')
    centroids = []
    B = data.shape[0]
    if B > nmax:
        data = data[:nmax, :]
        B = nmax
    for m in range(M):
        if verbose:
            print('Choosing Centroid %s...' % m)
        if m == 0:
            centroids.append(np.random.randint(0, high=B))
        else:
            dist = np.array([(data != data[centroid, :][np.newaxis]).mean(-1)
                             for centroid in centroids]).min(0)
            p = dist / dist.sum()
            if verbose:
                print('Distance computed, starting tower sampling')
            centroids.append(tower_sampling(1, np.cumsum(p))[0])
            if verbose:
                print('Done')
    return centroids


class MoI:
    def __init__(self, N=10, M=5, n_c=2, nature='Potts', gauge='zerosum', random_state=None):
        self.N = N
        self.M = M
        self.n_c = n_c
        self.gauge = gauge
        self.nature = nature
        self.muh = np.ones(M, dtype=curr_float) / M
        self.cum_muh = np.cumsum(self.muh)
        self.gh = np.zeros(M, dtype=curr_float)
        if nature == 'Potts':
            self.weights = np.zeros([M, N, n_c], dtype=curr_float)
        else:
            self.weights = np.zeros([M, N], dtype=curr_float)
        if nature == 'Bernoulli':
            self.cond_muv = np.ones([M, N], dtype=curr_float) / 2
        elif nature == 'Spin':
            self.cond_muv = np.zeros([M, N], dtype=curr_float)
        elif nature == 'Potts':
            self.cond_muv = np.ones([M, N, n_c], dtype=curr_float) / n_c
            self.cum_cond_muv = np.cumsum(self.cond_muv, axis=-1)

        self.random_state = utilities.check_random_state(random_state)
        self.logpartition()

    def logpartition(self):
        if self.nature == 'Bernoulli':
            self.logZ = np.logaddexp(0, self.weights).sum(-1)
        elif self.nature == 'Spin':
            self.logZ = np.logaddexp(self.weights, -self.weights).sum(-1)
        elif self.nature == 'Potts':
            self.logZ = utilities.logsumexp(self.weights, axis=-1).sum(-1)

    def likelihood(self, data):
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if self.nature in ['Bernoulli', 'Spin']:
            f = np.dot(data, self.weights.T)
        elif self.nature == 'Potts':
            f = cy_utilities.compute_output_C(data, self.weights, np.zeros(
                [data.shape[0], self.M], dtype=curr_float))
        return utilities.logsumexp((f - self.logZ[np.newaxis, :] + np.log(self.muh)[np.newaxis, :]), axis=1)

    def likelihood_and_expectation(self, data):
        if self.nature in ['Bernoulli', 'Spin']:
            f = np.dot(data, self.weights.T)
        elif self.nature == 'Potts':
            f = cy_utilities.compute_output_C(data, self.weights, np.zeros(
                [data.shape[0], self.M], dtype=curr_float))
        L = utilities.logsumexp(
            (f - self.logZ[np.newaxis, :] + np.log(self.muh)[np.newaxis, :]), axis=1)
        cond_muh = np.exp(
            f - self.logZ[np.newaxis, :]) * self.muh[np.newaxis, :]
        cond_muh /= cond_muh.sum(-1)[:, np.newaxis]
        return L, cond_muh

    def symKL(self, PGM, data, weights=None):
        n_samples = data.shape[0]
        data_moi, _ = self.gen_data(n_samples)
        D = -utilities.average(self.likelihood(data) + PGM.free_energy(data), weights=weights) + (
            self.likelihood(data_moi) + PGM.free_energy(data_moi)).mean()
        D /= self.N
        return D

    def gen_data(self, Ndata):
        h = tower_sampling(Ndata, self.cum_muh)
        rng = self.random_state.rand(Ndata, self.N)
        if self.nature == 'Bernoulli':
            v = (rng < self.cond_muv[h, :])
        elif self.nature == 'Spin':
            v = 2 * (rng < (1 + self.cond_muv[h, :]) / 2) - 1
        elif self.nature == 'Potts':
            v = cy_utilities.tower_sampling_C(
                Ndata, self.N, self.n_c, self.cum_cond_muv[h, :, :], rng)
        return v, h

    def expectation(self, data):
        if data.ndim == 1:
            data = data[np.newaxis, :]

        if self.nature in ['Bernoulli', 'Spin']:
            f = np.dot(data, self.weights.T)
        elif self.nature == 'Potts':
            f = cy_utilities.compute_output_C(data, self.weights, np.zeros(
                [data.shape[0], self.M], dtype=curr_float))

        tmp = f - self.logZ[np.newaxis, :]
        tmp -= tmp.max(-1)[:, np.newaxis]
        cond_muh = np.exp(tmp) * self.muh[np.newaxis, :]
        cond_muh /= cond_muh.sum(-1)[:, np.newaxis]
        return cond_muh

    def maximization(self, data, cond_muh, weights=None, eps=1e-6):
        self.muh = utilities.average(cond_muh, weights=weights)
        self.cum_muh = np.cumsum(self.muh)
        self.gh = np.log(self.muh)
        self.gh -= self.gh.mean()
        if self.nature == 'Bernoulli':
            self.cond_muv = utilities.average_product(
                cond_muh, data, mean1=True, weights=weights) / self.muh[:, np.newaxis]
            self.weights = np.log((self.cond_muv + eps) /
                                  (1 - self.cond_muv + eps))
        elif self.nature == 'Spin':
            self.cond_muv = utilities.average_product(
                cond_muh, data, mean1=True, weights=weights) / self.muh[:, np.newaxis]
            self.weights = 0.5 * \
                np.log((1 + self.cond_muv + eps) / (1 - self.cond_muv + eps))

        elif self.nature == 'Potts':
            self.cond_muv = utilities.average_product(
                cond_muh, data, c2=self.n_c, mean1=True, weights=weights) / self.muh[:, np.newaxis, np.newaxis]
            self.cum_cond_muv = np.cumsum(self.cond_muv, axis=-1)
            self.weights = np.log(self.cond_muv + eps)
            self.weights -= self.weights.mean(-1)[:, :, np.newaxis]
        self.logpartition()

    def minibatch_fit_symKL(self, data_PGM, PGM=None, data_MOI=None, F_PGM_dPGM=None, F_PGM_dMOI=None, F_MOI_dPGM=None, F_MOI_dMOI=None, cond_muh_dPGM=None, cond_muh_dMOI=None, weights=None):
        if data_MOI is None:
            data_MOI, _ = self.gen_data(data_PGM.shape[0])
        if F_PGM_dPGM is None:
            F_PGM_dPGM = PGM.free_energy(data_PGM)
        if F_PGM_dMOI is None:
            F_PGM_dMOI = PGM.free_energy(data_MOI)
        if (F_MOI_dPGM is None) | (cond_muh_dPGM is None):
            F_MOI_dPGM, cond_muh_dPGM = self.likelihood_and_expectation(
                data_PGM)
            F_MOI_dPGM *= -1
        if (F_MOI_dMOI is None) | (cond_muh_dMOI is None):
            F_MOI_dMOI, cond_muh_dMOI = self.likelihood_and_expectation(
                data_MOI)
            F_MOI_dMOI *= -1

        delta_lik = -F_PGM_dMOI + F_MOI_dMOI
        delta_lik -= delta_lik.mean()

        self.gradient = {}
        self.gradient['gh'] = utilities.average(
            cond_muh_dPGM, weights=weights) - self.muh + (delta_lik[:, np.newaxis] * cond_muh_dMOI).mean(0)
        if self.nature in ['Bernoulli', 'Spin']:
            self.gradient['weights'] = utilities.average_product(
                cond_muh_dPGM, data_PGM, mean1=True, weights=weights) + utilities.average_product(cond_muh_dMOI * delta_lik[:, np.newaxis], data_MOI, mean1=True)
            self.gradient['weights'] -= self.muh[:, np.newaxis] * self.cond_muv
        elif self.nature == 'Potts':
            self.gradient['weights'] = utilities.average_product(cond_muh_dPGM, data_PGM, mean1=True, c2=self.n_c, weights=weights) + utilities.average_product(
                cond_muh_dMOI * delta_lik[:, np.newaxis], data_MOI, mean1=True, c2=self.n_c)
            self.gradient['weights'] -= self.muh[:,
                                                 np.newaxis, np.newaxis] * self.cond_muv

        self.gh += self.learning_rate * self.gradient['gh']
        self.weights += self.learning_rate * self.gradient['weights']

        self.muh = np.exp(self.gh)
        self.muh /= self.muh.sum()
        self.cum_muh = np.cumsum(self.muh)
        if self.nature == 'Bernoulli':
            self.cond_muv = utilities.logistic(self.weights)
        elif self.nature == 'Spin':
            self.cond_muv = np.tanh(self.weights)
        elif self.nature == 'Potts':
            self.weights -= self.weights.mean(-1)[:, :, np.newaxis]
            self.cond_muv = np.exp(self.weights)
            self.cond_muv /= self.cond_muv.sum(-1)[:, :, np.newaxis]
            self.cum_cond_muv = np.cumsum(self.cond_muv, axis=-1)
        self.logpartition()

    def fit(self, data, weights=None, init_bias=0.1, verbose=1, eps=1e-5, maxiter=100, split_merge=True):
        # B = data.shape[0]
        initial_centroids = KMPP_choose_centroids(
            data, self.M, verbose=verbose)
        # initial_centroids = np.argsort(np.random.rand(B))[:self.M]
        if self.nature == 'Bernoulli':
            self.weights += init_bias / self.N * \
                (data[initial_centroids] - 0.5)
        elif self.nature == 'Spin':
            self.weights += 0.25 * init_bias / self.N * data[initial_centroids]
        elif self.nature == 'Potts':
            self.weights += init_bias / self.N * \
                binarize(data[initial_centroids], self.n_c) - \
                init_bias / (self.n_c * self.N)

        n_epoch = 0
        converged = (n_epoch >= maxiter)  # if nothing...
        previous_L = utilities.average(
            self.likelihood(data), weights=weights) / self.N
        current_L = previous_L.copy()

        if self.M < 3:
            split_merge = False

        if verbose:
            print('Iteration 0, L = %.3f' % current_L)

        while not converged:
            cond_muh = self.expectation(data)
            self.maximization(data, cond_muh, weights=weights)
            previous_L = current_L.copy()
            current_L = utilities.average(
                self.likelihood(data), weights=weights) / self.N
            n_epoch += 1
            converged = (n_epoch >= maxiter) | (
                np.abs(current_L - previous_L) < eps)
            if verbose:
                print('Iteration %s, L = %.3f' % (n_epoch, current_L))

        if split_merge:
            converged2 = False
            while not converged2:
                current_weights = self.weights.copy()
                current_cond_muv = self.cond_muv.copy()
                current_gh = self.gh.copy()
                current_muh = self.muh.copy()
                # current_cum_muh = self.cum_muh.copy()
                current_logZ = self.logZ.copy()
                if self.nature == 'Potts':
                    current_cum_cond_muv = self.cum_cond_muv.copy()
                previous_L = current_L.copy()

                current_cond_muh = self.expectation(data)
                proposed_merge_splits = self.split_merge_criterion(
                    data, Cmax=5, weights=weights)
                for proposed_merge_split in proposed_merge_splits:
                    self.merge_split(proposed_merge_split)
                    proposed_L = self.partial_EM(data, current_cond_muh[:, proposed_merge_split].sum(
                        -1), proposed_merge_split, weights=weights, eps=eps, maxiter=10, verbose=verbose)
                    converged3 = False
                    while not converged3:
                        cond_muh = self.expectation(data)
                        self.maximization(data, cond_muh, weights=weights)
                        previous_proposed_L = proposed_L.copy()
                        proposed_L = utilities.average(
                            self.likelihood(data), weights=weights) / self.N
                        n_epoch += 1
                        converged3 = (n_epoch >= maxiter) | (
                            np.abs(proposed_L - previous_proposed_L) < eps)
                    if proposed_L - current_L > eps:
                        current_L = proposed_L.copy()
                        if verbose:
                            print('Iteration %s, Split-Merge (%s,%s,%s) accepted, L = %.3f' % (
                                n_epoch, proposed_merge_split[0], proposed_merge_split[1], proposed_merge_split[2], current_L))
                        break
                    else:
                        self.weights = current_weights.copy()
                        self.cond_muv = current_cond_muv.copy()
                        self.gh = current_gh.copy()
                        self.muh = current_muh.copy()
                        self.cum_muh = self.cum_muh.copy()
                        self.logZ = current_logZ.copy()
                        if self.nature == 'Potts':
                            self.cum_cond_muv = current_cum_cond_muv.copy()

                        if verbose:
                            print('Iteration %s, Split-Merge (%s,%s,%s) denied, Proposed L = %.3f' % (
                                n_epoch, proposed_merge_split[0], proposed_merge_split[1], proposed_merge_split[2], proposed_L))
                converged2 = (np.abs(current_L - previous_L) <
                              eps) | (n_epoch >= 2 * maxiter)
        return current_L

    def fit_symKL(self, PGM, data, weights=None, batch_size=100, learning_rate=0.1, lr_final=None, n_iter=10, lr_decay=True, decay_after=0.5, verbose=1, init='ML', shuffle_data=True, print_every=5, init_bias=0.01):
        n_samples = data.shape[0]
        n_batches = int(np.ceil(float(n_samples) / batch_size))
        batch_slices = list(utilities.gen_even_slices(n_batches * batch_size,
                                                      n_batches, n_samples))

        # learning_rate_init = copy.copy(learning_rate)
        self.learning_rate = learning_rate
        if lr_decay:
            start_decay = n_iter * decay_after
            if lr_final is None:
                lr_final = 1e-2 * learning_rate

            decay_gamma = (float(lr_final) / float(learning_rate)
                           )**(1 / float(n_iter * (1 - decay_after)))

        if init == 'ML':
            print('~~~ Fitting first to data ~~~')
            self.fit(data, weights=weights, verbose=1)
        else:
            B = data.shape[0]
            initial_centroids = np.argsort(np.random.rand(B))[:self.M]
            if self.nature == 'Bernoulli':
                self.weights += init_bias * (data[initial_centroids] - 0.5)
            elif self.nature == 'Spin':
                self.weights += 0.25 * init_bias * data[initial_centroids]
            elif self.nature == 'Potts':
                self.weights += init_bias * \
                    binarize(data[initial_centroids], self.n_c) - \
                    init_bias / self.n_c

        if shuffle_data:
            if weights is not None:
                permute = np.arange(data.shape[0])
                self.random_state.shuffle(permute)
                weights = weights[permute]
                data = data[permute, :]
            else:
                self.random_state.shuffle(data)
        if verbose:
            print('Epoch 0: Dist = %.4f' %
                  self.symKL(PGM, data, weights=weights))

        for epoch in range(1, n_iter + 1):
            if verbose:
                begin = time.time()
                print('Starting epoch %s' % epoch)
            if lr_decay:
                if (epoch > start_decay):
                    self.learning_rate *= decay_gamma

            for batch_slice in batch_slices:
                if weights is None:
                    data_mini = data[batch_slice]
                    weights_mini = None
                else:
                    data_mini = data[batch_slice]
                    weights_mini = weights[batch_slice]
                self.minibatch_fit_symKL(
                    data_mini, PGM=PGM, weights=weights_mini)
            if verbose:
                t = time.time() - begin
                if epoch % print_every == 0:
                    print('Finished epoch %s: time =%.2f s, Dist = %.4f' %
                          (epoch, t, self.symKL(PGM, data, weights=weights)))

            if shuffle_data:
                if weights is not None:
                    permute = np.arange(data.shape[0])
                    self.random_state.shuffle(permute)
                    weights = weights[permute]
                    data = data[permute, :]
                else:
                    self.random_state.shuffle(data)

    def fit_online(self, data, weights=None, batch_size=100, learning_rate=0.01, lr_final=None, n_iter=10, lr_decay=True, decay_after=0.5, verbose=1, shuffle_data=True, print_every=5, init_bias=0.001, init=None):
        n_samples = data.shape[0]
        n_batches = int(np.ceil(float(n_samples) / batch_size))
        batch_slices = list(utilities.gen_even_slices(n_batches * batch_size,
                                                      n_batches, n_samples))

        # learning_rate_init = copy.copy(learning_rate)
        self.learning_rate = learning_rate
        if lr_decay:
            start_decay = n_iter * decay_after
            if lr_final is None:
                lr_final = 1e-2 * learning_rate

            decay_gamma = (float(lr_final) / float(learning_rate)
                           )**(1 / float(n_iter * (1 - decay_after)))

        B = data.shape[0]
        if not init == 'previous':
            # initial_centroids = KMPP_choose_centroids(data,self.M)
            initial_centroids = np.argsort(np.random.rand(B))[:self.M]
            if self.nature == 'Bernoulli':
                self.weights += init_bias * (data[initial_centroids] - 0.5)
            elif self.nature == 'Spin':
                self.weights += 0.25 * init_bias * data[initial_centroids]
            elif self.nature == 'Potts':
                self.weights += init_bias * \
                    binarize(data[initial_centroids], self.n_c) - \
                    init_bias / self.n_c

            if self.nature == 'Bernoulli':
                self.muvh = np.ones(
                    [self.M, self.N], dtype=curr_float) / (2.0 * self.M)
            elif self.nature == 'Spin':
                self.muvh = np.zeros([self.M, self.N], dtype=curr_float)
            else:
                self.muvh = np.ones(
                    [self.M, self.N, self.n_c], dtype=curr_float) / (self.n_c * self.M)
        else:
            if not hasattr(self, 'muvh'):
                if self.nature == 'Potts':
                    self.muvh = self.cond_muv * \
                        self.muh[:, np.newaxis, np.newaxis]
                else:
                    self.muvh = self.cond_muv * self.muh[:, np.newaxis]

        if shuffle_data:
            if weights is not None:
                permute = np.arange(data.shape[0])
                self.random_state.shuffle(permute)
                weights = weights[permute]
                data = data[permute, :]
            else:
                self.random_state.shuffle(data)
        if verbose:
            print('Epoch 0: Lik = %.4f' % (utilities.average(
                self.likelihood(data), weights=weights) / self.N))

        for epoch in range(0, n_iter + 1):
            if verbose:
                begin = time.time()
                print('Starting epoch %s' % epoch)

            if epoch == 0:
                update = False
            else:
                update = True

            if lr_decay:
                if (epoch > start_decay):
                    self.learning_rate *= decay_gamma

            for batch_slice in batch_slices:
                if weights is None:
                    data_mini = data[batch_slice]
                    weights_mini = None
                else:
                    data_mini = data[batch_slice]
                    weights_mini = weights[batch_slice]
                self.minibatch_fit(
                    data_mini, weights=weights_mini, update=update)
            if verbose:
                t = time.time() - begin
                if epoch % print_every == 0:
                    print('Finished epoch %s: time =%.2f s, Lik = %.4f' % (
                        epoch, t, utilities.average(self.likelihood(data), weights=weights) / self.N))

            if shuffle_data:
                if weights is not None:
                    permute = np.arange(data.shape[0])
                    self.random_state.shuffle(permute)
                    weights = weights[permute]
                    data = data[permute, :]
                else:
                    self.random_state.shuffle(data)

    def minibatch_fit(self, data, weights=None, eps=1e-5, update=True):
        h = self.expectation(data)
        self.muh = self.learning_rate * \
            utilities.average(h, weights=weights) + \
            (1 - self.learning_rate) * self.muh
        self.cum_muh = np.cumsum(self.muh)
        if update:
            self.gh = np.log(self.muh + eps)
            self.gh -= self.gh.mean()
        if self.nature == 'Bernoulli':
            self.muvh = self.learning_rate * \
                utilities.average_product(
                    h, data, weights=weights) + (1 - self.learning_rate) * self.muvh
            if update:
                self.cond_muv = self.muvh / (self.muh[:, np.newaxis])
                self.weights = np.log(
                    (self.cond_muv + eps) / (1 - self.cond_muv + eps))
        elif self.nature == 'Spin':
            self.muvh = self.learning_rate * \
                utilities.average(h, data, weights=weights) + \
                (1 - self.learning_rate) * self.muvh
            if update:
                self.cond_muv = self.muvh / self.muh[:, np.newaxis]
                self.weights = 0.5 * \
                    np.log((1 + self.cond_muv + eps) /
                           (1 - self.cond_muv + eps))
        else:
            self.muvh = self.learning_rate * utilities.average_product(
                h, data, c2=self.n_c, weights=weights) + (1 - self.learning_rate) * self.muvh
            if update:
                self.cond_muv = self.muvh / self.muh[:, np.newaxis, np.newaxis]
                self.weights = np.log(self.cond_muv + eps)
                self.weights -= self.weights.mean(-1)[:, :, np.newaxis]

        if update:
            self.logpartition()

    def split_merge_criterion(self, data, Cmax=5, weights=None):
        likelihood, cond_muh = self.likelihood_and_expectation(data)
        norm = np.sqrt(utilities.average(cond_muh**2, weights=weights))
        J_merge = utilities.average_product(
            cond_muh, cond_muh, weights=weights) / (1e-10 + norm[np.newaxis, :] * norm[:, np.newaxis])
        J_merge = np.triu(J_merge, 1)
        proposed_merge = np.argsort(J_merge.flatten())[::-1][:Cmax]
        proposed_merge = [(merge % self.M, merge // self.M)
                          for merge in proposed_merge]

        tmp = cond_muh / self.muh[np.newaxis, :]

        if weights is None:
            J_split = np.array(
                [utilities.average(likelihood, weights=tmp[:, m]) for m in range(self.M)])
        else:
            J_split = np.array([utilities.average(
                likelihood, weights=tmp[:, m] * weights) for m in range(self.M)])

        proposed_split = np.argsort(J_split)[:3]
        proposed_merge_split = []
        for merge1, merge2 in proposed_merge:
            if proposed_split[0] in [merge1, merge2]:
                if proposed_split[1] in [merge1, merge2]:
                    proposed_merge_split.append(
                        (merge1, merge2, proposed_split[2]))
                else:
                    proposed_merge_split.append(
                        (merge1, merge2, proposed_split[1]))
            else:
                proposed_merge_split.append(
                    (merge1, merge2, proposed_split[0]))
        return proposed_merge_split

    def partial_EM(self, data, cond_muh_ijk, indices, weights=None, eps=1e-4, maxiter=10, verbose=0):
        (i, j, k) = indices
        converged = False
        previous_L = utilities.average(
            self.likelihood(data), weights=weights) / self.N
        mini_epochs = 0
        if verbose:
            print('Partial EM %s, L = %.3f' % (mini_epochs, previous_L))
        while not converged:
            if self.nature in ['Bernoulli', 'Spin']:
                f = np.dot(data, self.weights[[i, j, k], :].T)
            elif self.nature == 'Potts':
                f = cy_utilities.compute_output_C(data, self.weights[[i, j, k], :, :], np.zeros([
                                                  data.shape[0], 3], dtype=curr_float))

            tmp = f - self.logZ[np.newaxis, [i, j, k]]
            tmp -= tmp.max(-1)[:, np.newaxis]
            cond_muh = np.exp(tmp) * self.muh[np.newaxis, [i, j, k]]
            cond_muh /= cond_muh.sum(-1)[:, np.newaxis]
            cond_muh *= cond_muh_ijk[:, np.newaxis]

            self.muh[[i, j, k]] = utilities.average(cond_muh, weights=weights)
            self.cum_muh = np.cumsum(self.muh)
            self.gh[[i, j, k]] = np.log(self.muh[[i, j, k]])
            self.gh -= self.gh.mean()
            if self.nature == 'Bernoulli':
                self.cond_muv[[i, j, k]] = utilities.average_product(
                    cond_muh, data, mean1=True, weights=weights) / self.muh[[i, j, k], np.newaxis]
                self.weights[[i, j, k]] = np.log(
                    (self.cond_muv[[i, j, k]] + eps) / (1 - self.cond_muv[[i, j, k]] + eps))
                self.logZ[[i, j, k]] = np.logaddexp(
                    0, self.weights[[i, j, k]]).sum(-1)
            elif self.nature == 'Spin':
                self.cond_muv[[i, j, k]] = utilities.average_product(
                    cond_muh, data, mean1=True, weights=weights) / self.muh[[i, j, k], np.newaxis]
                self.weights[[i, j, k]] = 0.5 * np.log(
                    (1 + self.cond_muv[[i, j, k]] + eps) / (1 - self.cond_muv[[i, j, k]] + eps))
                self.logZ[[i, j, k]] = np.logaddexp(
                    self.weights[[i, j, k]], -self.weights[[i, j, k]]).sum(-1)
            elif self.nature == 'Potts':
                self.cond_muv[[i, j, k]] = utilities.average_product(
                    cond_muh, data, c2=self.n_c, mean1=True, weights=weights) / self.muh[[i, j, k], np.newaxis, np.newaxis]
                self.cum_cond_muv[[i, j, k]] = np.cumsum(
                    self.cond_muv[[i, j, k]], axis=-1)
                self.weights[[i, j, k]] = np.log(
                    self.cond_muv[[i, j, k]] + eps)
                self.weights[[i, j, k]] -= self.weights[[i, j, k]
                                                        ].mean(-1)[:, :, np.newaxis]
                self.logZ[[i, j, k]] = utilities.logsumexp(
                    self.weights[[i, j, k]], axis=-1).sum(-1)

            current_L = utilities.average(
                self.likelihood(data), weights=weights) / self.N
            mini_epochs += 1
            converged = (mini_epochs >= maxiter) | (
                np.abs(current_L - previous_L) < eps)
            previous_L = current_L.copy()
            if verbose:
                print('Partial EM %s, L = %.3f' % (mini_epochs, current_L))
        return current_L

    def merge_split(self, proposed_merge_split, eps=1e-6):
        i, j, k = proposed_merge_split
        old_mui = self.muh[i].copy()
        old_muj = self.muh[j].copy()
        old_muk = self.muh[k].copy()
        self.muh[i] = old_mui + old_muj
        self.muh[k] = old_muk / 2
        self.muh[j] = old_muk / 2
        self.gh = np.log(self.muh)
        self.gh -= self.gh.mean()
        self.cum_muh = np.cumsum(self.muh)

        old_cond_muvi = self.cond_muv[i].copy()
        old_cond_muvj = self.cond_muv[j].copy()
        old_cond_muvk = self.cond_muv[k].copy()

        self.cond_muv[i] = (old_cond_muvi * old_mui +
                            old_cond_muvj * old_muj) / (old_mui + old_muj)

        if self.nature == 'Potts':
            noise = np.random.rand(self.N, self.n_c)
            noise /= noise.sum(-1)[:, np.newaxis]
        elif self.nature == 'Bernoulli':
            noise = np.random.rand(self.N)
            noise /= noise.sum(-1)
        elif self.nature == 'Spin':
            noise = (2 * np.random.rand(self.N) - 1)

        self.cond_muv[j] = 0.95 * old_cond_muvk + 0.05 * noise

        if self.nature == 'Bernoulli':
            self.weights[[i, j]] = np.log(
                (self.cond_muv[[i, j]] + eps) / (1 - self.cond_muv[[i, j]] + eps))
            self.logZ[[i, j]] = np.logaddexp(0, self.weights[[i, j]]).sum(-1)
        elif self.nature == 'Spin':
            self.weights[[i, j]] = 0.5 * np.log(
                (1 + self.cond_muv[[i, j]] + eps) / (1 - self.cond_muv[[i, j]] + eps))
            self.logZ[[i, j]] = np.logaddexp(
                self.weights[[i, j]], -self.weights[[i, j]]).sum(-1)
        elif self.nature == 'Potts':
            self.cum_cond_muv[[i, j]] = np.cumsum(
                self.cond_muv[[i, j]], axis=-1)
            self.weights[[i, j]] = np.log(self.cond_muv[[i, j]] + eps)
            self.weights[[i, j]] -= self.weights[[i, j]
                                                 ].mean(-1)[:, :, np.newaxis]
            self.logZ[[i, j]] = utilities.logsumexp(
                self.weights[[i, j]], axis=-1).sum(-1)
