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

import layer
import numpy as np
from float_precision import double_precision, curr_float, curr_int
import numba_utilities
import copy
import time
import utilities


class GLM:
    def __init__(self, Nin=100, Nout=100, n_cin=1, n_cout=1, nature='Gaussian',symmetric=False,zero_diag=False):
        self.nature = nature
        self.Nin = Nin
        self.Nout = Nout
        self.n_cin = n_cin
        self.n_cout = n_cout
        self.symmetric = symmetric
        if self.symmetric:
            assert self.Nin == self.Nout
            assert self.n_cin == self.n_cout
        self.zero_diag = zero_diag
        if self.zero_diag:
            assert self.Nin == self.Nout
            assert self.n_cin == self.n_cout

        if n_cin > 1:
            nature_in = 'Potts'
        else:
            nature_in = 'Gaussian'
        self.input_layer = layer.initLayer(nature=nature_in, N=Nin, n_c=n_cin)
        self.output_layer = layer.initLayer(nature=nature, N=Nout, n_c=n_cout)
        if (n_cin > 1) & (n_cout > 1):
            self.weights = np.zeros(
                [Nin, Nout, n_cin, n_cout], dtype=curr_float)
        elif (n_cin == 1) & (n_cout > 1):
            self.weights = np.zeros([Nin, Nout, n_cout], dtype=curr_float)
        else:
            self.weights = np.zeros([Nin, Nout], dtype=curr_float)

    def predict(self, X):
        X = X.astype(self.input_layer.type)
        return self.output_layer.mean_from_inputs(self.input_layer.compute_output(X, self.weights, direction='down'))

    def predict_samples(self, X, n_samples=50):
        X = X.astype(self.input_layer.type)
        I = self.input_layer.compute_output(
            X, self.weights, direction='down')
        samples = np.swapaxes(np.array(
            [self.output_layer.sample_from_inputs(I) for _ in range(n_samples)]), 0, 1)
        return samples

    def likelihood(self, X, Y):
        X = X.astype(self.input_layer.type)
        Y = Y.astype(self.output_layer.type)
        I = self.input_layer.compute_output(
            X, self.weights, direction='down')
        L = -self.output_layer.energy(Y) - self.output_layer.logpartition(I)
        if self.nature == 'Potts':
            L += numba_utilities.dot_Potts2_C(Y, I)
        else:
            L += (Y * I).sum(1)
        return L / self.Nout

    def minibatch_fit(self, X, Y, weights=None):
        self.count_updates += 1
        grad = {}
        I = self.input_layer.compute_output(X, self.weights, direction='down')

        prediction = self.output_layer.mean_from_inputs(I)
        grad['weights'] = self.moments_XY - utilities.average_product(
            X, prediction, c1=self.n_cin, c2=self.n_cout, mean2=True, weights=weights)
        if self.nature in ['Bernoulli','Spin','Potts']:
            grad['output_layer'] = self.output_layer.internal_gradients(
                self.moments_Y, prediction, value='moments', value_neg='mean', weights_neg=weights)
        else:
            grad['output_layer'] = self.output_layer.internal_gradients(
                self.moments_Y, I, value='moments', value_neg='input', weights_neg=weights)

        for regtype, regtarget, regvalue in self.regularizers:
            if regtarget == 'weights':
                target_gradient = grad['weights']
                target = self.weights
            else:
                target_gradient = grad['output_layer'][regtarget]
                target = self.output_layer.__dict__[regtarget]
            if regtype == 'l1':
                target_gradient -= regvalue * np.sign(target)
            elif regtype == 'l2':
                target_gradient -= regvalue * target
            else:
                print(regtype, 'not supported')

        for key, gradient in grad['output_layer'].items():
            if self.output_layer.do_grad_updates[key]:
                if self.optimizer == 'SGD':
                    self.output_layer.__dict__[
                        key] += self.learning_rate * gradient
                elif self.optimizer == 'ADAM':
                    self.gradient_moment1['output_layer'][key] *= self.beta1
                    self.gradient_moment1['output_layer'][key] += (
                        1 - self.beta1) * gradient
                    self.gradient_moment2['output_layer'][key] *= self.beta2
                    self.gradient_moment2['output_layer'][key] += (
                        1 - self.beta2) * gradient**2

                    self.output_layer.__dict__[key] += self.learning_rate / (1 - self.beta1) * (self.gradient_moment1['output_layer'][key] / (
                        1 - self.beta1**self.count_updates)) / (self.epsilon + np.sqrt(self.gradient_moment2['output_layer'][key] / (1 - self.beta2**self.count_updates)))

        if self.optimizer == 'SGD':
            self.weights += self.learning_rate * grad['weights']
        elif self.optimizer == 'ADAM':
            self.gradient_moment1['weights'] *= self.beta1
            self.gradient_moment1['weights'] += (1 -
                                                 self.beta1) * grad['weights']
            self.gradient_moment2['weights'] *= self.beta2
            self.gradient_moment2['weights'] += (1 -
                                                 self.beta2) * grad['weights']**2

            self.weights += self.learning_rate / (1 - self.beta1) * (self.gradient_moment1['weights'] / (
                1 - self.beta1**self.count_updates)) / (self.epsilon + np.sqrt(self.gradient_moment2['weights'] / (1 - self.beta2**self.count_updates)))

        if self.symmetric:
            if self.n_cout>1:
                self.weights += np.swapaxes(np.swapaxes(self.weights,0,1),2,3)
                self.weights /= 2
            else:
                self.weights += self.weights.T
                self.weights /=2
        if self.zero_diag:
            self.weights[np.arange(self.Nout),np.arange(self.Nout)] *= 0
        return

    def fit(self, X, Y, weights=None, batch_size=100, learning_rate=None, lr_final=None, lr_decay=True, decay_after=0.5,
            extra_params=None, optimizer='ADAM', n_iter=10, verbose=1, regularizers=[]):

        self.batch_size = batch_size
        self.optimizer = optimizer
        self.n_iter = n_iter
        if self.n_iter <= 1:
            lr_decay = False
        if learning_rate is None:
            if self.optimizer == 'SGD':
                learning_rate = 0.01
            elif self.optimizer == 'ADAM':
                learning_rate = 5e-4
            else:
                print('Need to specify learning rate for optimizer.')
        if self.optimizer == 'ADAM':
            if extra_params is None:
                extra_params = [0.9, 0.99, 1e-3]
            self.beta1 = extra_params[0]
            self.beta2 = extra_params[1]
            self.epsilon = extra_params[2]
            if self.n_cout > 1:
                out0 = np.zeros([1, self.Nout, self.n_cout], dtype=curr_float)
            else:
                out0 = np.zeros([1, self.Nout], dtype=curr_float)

            grad = {'weights': np.zeros_like(self.weights),
                    'output_layer': self.output_layer.internal_gradients(
                out0, out0, value='input', value_neg='input')}

            for key in grad['output_layer'].keys():
                grad['output_layer'][key] *= 0

            self.gradient_moment1 = copy.deepcopy(grad)
            self.gradient_moment2 = copy.deepcopy(grad)

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
        self.regularizers = regularizers

        n_samples = X.shape[0]
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(utilities.gen_even_slices(n_batches * self.batch_size,
                                                      n_batches, n_samples))

        X = np.asarray(X,dtype=self.input_layer.type,order='c')
        Y = np.asarray(Y,dtype=self.output_layer.type,order='c')
        if weights is not None:
            weights = weights.astype(curr_float)

        self.moments_Y = self.output_layer.get_moments(
            Y, weights=weights, value='data')
        self.moments_XY = utilities.average_product(
            X, Y, c1=self.n_cin, c2=self.n_cout, mean1=False, mean2=False, weights=weights)

        self.count_updates = 0

        for epoch in range(1, n_iter + 1):
            if verbose:
                begin = time.time()
            if self.lr_decay:
                if (epoch > self.start_decay):
                    self.learning_rate *= self.decay_gamma

            permutation = np.argsort(np.random.randn(n_samples))
            X = X[permutation, :]
            Y = Y[permutation, :]
            if weights is not None:
                weights = weights[permutation]

            if verbose:
                print('Starting epoch %s' % (epoch))
            for batch_slice in batch_slices:
                if weights is not None:
                    self.minibatch_fit(
                        X[batch_slice], Y[batch_slice], weights=weights[batch_slice])
                else:
                    self.minibatch_fit(
                        X[batch_slice], Y[batch_slice], weights=None)

            if verbose:
                end = time.time()
                lik = utilities.average(self.likelihood(X, Y), weights=weights)
                regularization = 0
                for regtype, regtarget, regvalue in self.regularizers:
                    if regtarget == 'weights':
                        target = self.weights
                    else:
                        target = self.output_layer.__dict__[regtarget]
                    if regtype == 'l1':
                        regularization += (regvalue * np.abs(target)).sum()
                    elif regtype == 'l2':
                        regularization += 0.5 * (regvalue * target**2).sum()
                    else:
                        print(regtype, 'not supported')
                    regularization /= self.Nout
                message = "Iteration %d, time = %.2fs, likelihood = %.2f, regularization  = %.2e, loss = %.2f" % (
                    epoch, end - begin, lik, regularization, -lik + regularization)
                print(message)
        return 'done'
