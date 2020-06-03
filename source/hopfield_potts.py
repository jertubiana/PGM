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

import scipy
import utilities
import numpy as np
import bm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



class HopfieldPotts(bm.BM):
    def __init__(self,  N = 100, M=10, n_c=21):
        self.M = M
        self.xi = np.zeros([M,N,n_c])
        self.lam = np.zeros(M)
        self.DeltaL = np.zeros(M)

        super(HopfieldPotts,self).__init__(N=N,n_c=n_c,nature='Potts')

    def fit(self, data,weights=None, pseudo_count = 1e-4,verbose=1,zero_diag=True):

        fi = utilities.average(data,c=self.n_c, weights = weights)
        fij = utilities.average_product(data,data,c1=self.n_c,c2=self.n_c, weights = weights)
        for i in range(self.N):
            fij[i,i] = np.diag(fi[i])

        fi_PC = (1-pseudo_count) * fi + pseudo_count/float(self.n_c)
        fij_PC = (1-pseudo_count) * fij + pseudo_count/float(self.n_c)**2

        for i in range(self.N):
            fij_PC[i,i] = np.diag(fi_PC[i])

        Cij = fij_PC - fi_PC[np.newaxis,:,np.newaxis,:] * fi_PC[:,np.newaxis,:,np.newaxis]


        D = np.zeros([self.N,self.n_c-1,self.n_c-1])
        invD = np.zeros([self.N,self.n_c-1,self.n_c-1])

        for n in range(self.N):
            D[n] = scipy.linalg.sqrtm( Cij[n,n,:-1,:-1])
            invD[n] = np.linalg.inv(D[n])

        Gamma = np.zeros([self.N,self.n_c-1,self.N,self.n_c-1])
        for n1 in range(self.N):
            for n2 in range(self.N):
                Gamma[n1,:,n2,:] = np.dot(invD[n1], np.dot( Cij[n1,n2,:-1,:-1],invD[n2]) )


        Gamma_bin = Gamma.reshape([self.N*(self.n_c-1),self.N*(self.n_c-1)])
        Gamma_bin = (Gamma_bin+Gamma_bin.T)/2
        lam, v = np.linalg.eigh(Gamma_bin)
        order = np.argsort(lam)[::-1]

        v_ordered = np.rollaxis(v.reshape([self.N,self.n_c-1,self.N*(self.n_c-1)]),2,0)[order,:,:]
        lam_ordered = lam[order]
        DeltaL = 0.5 * (lam_ordered-1 -  np.log(lam_ordered))
        xi = np.zeros(v_ordered.shape)
        for n in range(self.N):
            xi[:,n,:] = np.dot(v_ordered[:,n,:],invD[n])
        xi = np.sqrt( np.abs(1-1/lam_ordered) )[:,np.newaxis,np.newaxis] * xi


        xi = np.concatenate( (xi, np.zeros([self.N*(self.n_c-1),self.N,1]) ), axis=2 ) # Write in zero-sum gauge.
        xi -= xi.mean(-1)[:,:,np.newaxis]
        top_M_contrib = np.argsort(DeltaL)[::-1][:self.M]


        self.xi = xi[top_M_contrib]
        self.lam = lam_ordered[top_M_contrib]
        self.DeltaL = DeltaL[top_M_contrib]


        couplings =  np.tensordot(self.xi[self.lam>1], self.xi[self.lam>1], axes=[(0),(0)]) - np.tensordot(self.xi[self.lam<1], self.xi[self.lam<1], axes=[(0),(0)])
        couplings = np.asarray(np.swapaxes(couplings, 1,2),order='c')
        if zero_diag: # With zero diag is much better; I just check things...
            for n in range(self.N):
                couplings[n,n,:,:]*=0



        fields = np.log(fi_PC) - np.tensordot(couplings, fi_PC,axes=[(1,3),(0,1)] )
        fields -= fields.mean(-1)[:,np.newaxis]

        self.layer.couplings = couplings
        self.layer.fields = fields
        if verbose:
            fig, ax = plt.subplots()
            ax2 = ax.twinx()
            ax.plot(self.DeltaL)
            ax2.semilogy(self.lam,c='red')
            ax.set_ylabel(r'$\Delta L$',color='blue')
            ax2.set_ylabel('Mode variance',color='red')
            for tl in ax.get_yticklabels():
                tl.set_color('blue')
            for tl in ax2.get_yticklabels():
                tl.set_color('red')
