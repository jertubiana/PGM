"""
 Copyright 2018 - by Jerome Tubiana (jertubiana@@gmail.com)
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
import sys
import pickle
import os
sys.path.append('../source/')
import utilities
import copy
import types
from multiprocessing import Pool
from functools import partial
import rbm
import layer
from moi import KMPP_choose_centroids



def saveRBM(filename,RBM):
    pickle.dump(RBM,open(filename,'wb'))

def loadRBM(filename):
    return pickle.load(open(filename,'rb'))


def copyRBM(RBM,package=rbm,**extra_keys): # Useful when want to copy all attributes while chaging the methods.
    env = {}
    for key in ['interpolate','degree_interpolate','interpolate_z']:
        if key in extra_keys.keys():
            env[key] = extra_keys[key]
        else:
            env[key] = RBM.__dict__[key]

    copy = package.RBM(n_v=RBM.n_v,n_cv = RBM.n_cv, n_h=RBM.n_h, n_ch = RBM.n_ch,visible=RBM.visible, hidden = RBM.hidden, interpolate=env['interpolate'], degree_interpolate=env['degree_interpolate'])
    for key,item in RBM.vlayer.__dict__.items():
        copy.vlayer.__dict__[key] = item
    for key,item in RBM.hlayer.__dict__.items():
        copy.hlayer.__dict__[key] = item
    copy.weights = RBM.weights.copy()
    if hasattr(RBM,'zlayer'):
        if env['interpolate_z']:
            copy.zlayer = package.layer.initLayer(N=1,n_c= RBM.zlayer.n_c, nature='PottsInterpolate', degree = RBM.zlayer.degree)
        else:
            copy.zlayer = package.layer.initLayer(N=1,n_c= RBM.zlayer.n_c, nature='Potts', degree = RBM.zlayer.degree)
        for key,item in RBM.zlayer.__dict__.items():
            copy.zlayer.__dict__[key] = item

        copy.weights_MoI = RBM.weights_MoI.copy()
    if hasattr(RBM,'betas'):
        copy.betas = RBM.betas.copy()
    return copy

def load_old_RBM(RBM_old, python2='python',path_to_old_package='/Users/jerometubiana/Desktop/PhD/MSA_Potts_RBM/Code_for_Paper/RBM/',package=rbm):
    script_python2 = "\
import pickle;\
import sys;\
sys.path.append('%s');\
RBM = pickle.load(open('%s','r'));\
env = {'n_v':RBM.n_v, 'n_h':RBM.n_h,'n_cv':RBM.n_cv,'n_ch':RBM.n_ch,'visible':RBM.visible,'hidden':RBM.hidden,\
'vfields':RBM.vlayer.fields, 'vfields0':RBM.vlayer.fields0,\
'htheta_plus':RBM.hlayer.theta_plus,'hgamma_plus':RBM.hlayer.a_plus,\
'htheta_minus':RBM.hlayer.theta_minus,'hgamma_minus':RBM.hlayer.a_minus,\
'htheta_plus0':RBM.hlayer.theta_plus0,'hgamma_plus0':RBM.hlayer.a_plus0,\
'htheta_minus0':RBM.hlayer.theta_minus0,'hgamma_minus0':RBM.hlayer.a_minus0,\
'weights':RBM.weights};\
pickle.dump(env,open('tmp.data','wb'));"%(path_to_old_package,RBM_old)

    cmd = '%s -c "%s"'%(python2,script_python2)
    os.system(cmd)
    env = pickle.load(open('tmp.data','rb'),encoding='latin1')
    RBM = package.RBM(n_v=env['n_v'],n_h=env['n_h'],n_cv=env['n_cv'],n_ch=env['n_ch'],visible=env['visible'],hidden=env['hidden'])
    RBM.vlayer.fields = env['vfields'].astype(np.float32)
    RBM.vlayer.fields0 = env['vfields0'].astype(np.float32)
    RBM.hlayer.theta_plus = env['htheta_plus'].astype(np.float32)
    RBM.hlayer.theta_plus0 = env['htheta_plus0'].astype(np.float32)
    RBM.hlayer.theta_minus = env['htheta_minus'].astype(np.float32)
    RBM.hlayer.theta_minus0 = env['htheta_minus0'].astype(np.float32)
    RBM.hlayer.gamma_plus = env['hgamma_plus'].astype(np.float32)
    RBM.hlayer.gamma_plus0 = env['hgamma_plus0'].astype(np.float32)
    RBM.hlayer.gamma_minus = env['hgamma_minus'].astype(np.float32)
    RBM.hlayer.gamma_minus0 = env['hgamma_minus0'].astype(np.float32)
    RBM.weights = env['weights'].astype(np.float32)
    RBM.hlayer.recompute_params(which='other')
    os.system('rm %s'%RBM_old)
    return RBM


def get_norm(W,include_gaps=True,a=2):
    if not include_gaps:
        W_ = W[:,:,:-1]
    else:
        W_ = W
    return  ( (np.abs(W_)**a).sum(-1).sum(-1) )**(1./a)


def get_norm_gaps(W,a=2):
    return ( (np.abs(W[:,:,-1])**a).sum(-1) )**(1./a)

def get_sparsity(W,a=3,include_gaps=True):
    if not include_gaps:
        W_ = W[:,:,:-1]
    else:
        W_ = W
    tmp = np.sqrt((W_**2).sum(-1))
    p = ((tmp**a).sum(1))**2/(tmp**(2*a)).sum(1)
    return p/W.shape[1]

def get_hlayer_asymmetry(RBM):
    return np.abs(RBM.hlayer.eta)

def get_hlayer_jump(RBM,positive_only=True):
    if not RBM.hidden == 'dReLU':
        print('get_hlayer_jump not supported for hidden %s'%RBM.hidden)
    else:
        jump = -(RBM.hlayer.theta_plus+RBM.hlayer.theta_minus)/(np.sqrt(RBM.hlayer.gamma_plus*RBM.hlayer.gamma_minus))
        if positive_only:
            return np.maximum(jump , 0)
        else:
            return jump


def get_hidden_unit_importance(RBM,data,weights=None,Nchains=200,Nthermalize=1000,Nstep=10,Lchains=100,init='data',recompute=False):
    '''
    Estimate the hidden unit importance as the difference between the data likelihood of the full RBM and the data likelihood of the where hidden unit has been removed.
    Estimated by importance sampling (see K. Shimagaki, M. Weigt 2019 for a derivation for Gaussian variables).
    Delta L takes into account both the norm of the weights and the shape of the non-linearity.
    Compared to the weight norm, it tends to put lower importance to the “phylogenetic” features, activated by very few sequences.    
    '''
    if (not recompute) & hasattr(RBM,'hidden_unit_importance'):
        return RBM.hidden_unit_importance
    else:
        if init == 'data':
            h = RBM.mean_hiddens(data)
            initial_points = data[KMPP_choose_centroids(h,Nchains)]
        else:
            initial_points = []

        data_gen , _ = RBM.gen_data(Nthermalize=Nthermalize,Nchains=Nchains,Nstep=Nstep,Lchains=Lchains,config_init=initial_points)

        cgf = RBM.hlayer.cgf_from_inputs(RBM.input_hiddens(data))
        cgf_gen = RBM.hlayer.cgf_from_inputs(RBM.input_hiddens(data_gen))
        DeltaL = utilities.average(cgf,weights=weights) + utilities.logsumexp(-cgf_gen,axis=0) - np.log(len(data_gen))
        RBM.hidden_unit_importance = DeltaL
        return DeltaL


def get_hidden_input(data,RBM,normed=False,offset=True):
    if normed:
        mu = utilities.average(data,c=21)
        norm_null = np.sqrt(  ((RBM.weights**2 * mu).sum(-1) - (RBM.weights*mu).sum(-1)**2).sum(-1) )
        return (RBM.input_hiddens(data) - RBM.hlayer.mu_I)/norm_null[np.newaxis,:]
    else:
        if offset:
            return (RBM.vlayer.compute_output(data,RBM.weights) - RBM.hlayer.mu_I)
        else:
            return (RBM.vlayer.compute_output(data,RBM.weights) )



def conditioned_RBM(RBM, conditions):
    num_conditions = len(conditions)
    l_hh = np.array( [condition[0] for condition in conditions] )
    l_value = np.array( [condition[1] for condition in conditions] )

    remaining = np.array( [x for x in range(RBM.n_h) if not x in l_hh] )
    tmp_RBM = rbm.RBM(n_v = RBM.n_v, n_h = RBM.n_h-num_conditions, n_cv = RBM.n_cv, n_ch = RBM.n_ch, visible = RBM.visible, hidden= RBM.hidden, interpolate=RBM.interpolate)
    tmp_RBM.vlayer = copy.deepcopy(RBM.vlayer)
    tmp_RBM.vlayer.fields += (RBM.weights[l_hh] * l_value[:,np.newaxis,np.newaxis]).sum(0)
    tmp_RBM.vlayer.fields0 += (RBM.weights[l_hh] * l_value[:,np.newaxis,np.newaxis]).sum(0)
    tmp_RBM.weights = RBM.weights[remaining]

    for param_name in RBM.hlayer.list_params:
        tmp_RBM.hlayer.__dict__[param_name] = RBM.hlayer.__dict__[param_name][remaining].copy()
        if hasattr(RBM.hlayer,param_name+'0'):
            tmp_RBM.hlayer.__dict__[param_name+'0'] = RBM.hlayer.__dict__[param_name+'0'][remaining].copy()
        if hasattr(RBM.hlayer,param_name+'1'):
            tmp_RBM.hlayer.__dict__[param_name+'1'] = RBM.hlayer.__dict__[param_name+'1'][:,remaining].copy()
    return tmp_RBM



def gen_data_lowT(RBM, beta=1, which = 'marginal' ,**generation_options):
    if which == 'joint':
        tmp_RBM = copy.deepcopy(RBM)
        tmp_RBM.weights *= beta
        for param_name in tmp_RBM.vlayer.params:
            tmp_RBM.vlayer.__dict__[param_name] *= beta
        for param_name in tmp_RBM.hlayer.params:
            tmp_RBM.hlayer.__dict__[param_name] *= beta

    elif which == 'marginal':
        if type(beta) == int:
            tmp_RBM = rbm.RBM(n_v=RBM.n_v, n_h = beta* RBM.n_h,visible=RBM.visible,hidden=RBM.hidden, n_cv = RBM.n_cv, n_ch = RBM.n_ch,interpolate=RBM.interpolate)
            tmp_RBM.vlayer = copy.deepcopy(RBM.vlayer)
            for param_name in tmp_RBM.vlayer.list_params:
                tmp_RBM.vlayer.__dict__[param_name] *= beta
            tmp_RBM.weights = np.repeat(RBM.weights,beta,axis=0)
        for param_name in tmp_RBM.hlayer.list_params:
            tmp_RBM.hlayer.__dict__[param_name] = np.repeat(RBM.hlayer.__dict__[param_name],beta,axis=0)
            tmp_RBM.hlayer.__dict__[param_name] = np.repeat(RBM.hlayer.__dict__[param_name],beta,axis=0)
            if hasattr(RBM.hlayer,param_name+'0'):
                tmp_RBM.hlayer.__dict__[param_name+'0'] = np.repeat(RBM.hlayer.__dict__[param_name+'0'],beta,axis=0)
                tmp_RBM.hlayer.__dict__[param_name+'0'] = np.repeat(RBM.hlayer.__dict__[param_name+'0'],beta,axis=0)
            if hasattr(RBM.hlayer,param_name+'1'):
                tmp_RBM.hlayer.__dict__[param_name+'1'] = np.repeat(RBM.hlayer.__dict__[param_name+'1'],beta,axis=1)
                tmp_RBM.hlayer.__dict__[param_name+'1'] = np.repeat(RBM.hlayer.__dict__[param_name+'1'],beta,axis=1)
    return tmp_RBM.gen_data(**generation_options)



def gen_data_zeroT(RBM, which = 'marginal' ,Nchains=10,Lchains=100,Nthermalize=0,Nstep=1,N_PT=1,reshape=True,update_betas=False,config_init=[]):
    tmp_RBM = copy.deepcopy(RBM)
    if which == 'joint':
        tmp_RBM.markov_step = types.MethodType(markov_step_zeroT_joint, tmp_RBM)
    elif which == 'marginal':
        tmp_RBM.markov_step = types.MethodType(markov_step_zeroT_marginal, tmp_RBM)
    return tmp_RBM.gen_data(Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)



def markov_step_zeroT_joint(self,x,beta=1):
    (v,h) = x
    I = self.vlayer.compute_output(v,self.weights,direction='up')
    h = self.hlayer.transform(I)
    I = self.hlayer.compute_output(h,self.weights,direction='down')
    v = self.vlayer.transform(I)
    return (v,h)

def markov_step_zeroT_marginal(self,x,beta=1):
    (v,h) = x
    I = self.vlayer.compute_output(v,self.weights,direction='up')
    h = self.hlayer.mean_from_inputs(I)
    I = self.hlayer.compute_output(h,self.weights,direction='down')
    v = self.vlayer.transform(I)
    return (v,h)


def get_effective_couplings_approx(RBM,data,weights=None):
    I = RBM.input_hiddens(data)
    conditional_variance = RBM.hlayer.var_from_inputs(I)
    mean_var = utilities.average(conditional_variance,weights=weights)
    if RBM.n_cv>1:
        J_eff = np.tensordot( RBM.weights, RBM.weights * mean_var[:,np.newaxis,np.newaxis], axes= [0,0] )
        J_eff = np.swapaxes(J_eff,1,2)
    else:
        J_eff = np.dot(RBM.weights.T, RBM.weights * mean_var[:,np.newaxis])
    J_eff[np.arange(RBM.n_v),np.arange(RBM.n_v)] = 0 # Remove diagonal entries.
    return J_eff

def get_effective_couplings_exact(RBM,data,weights=None,nbins=10,subset = None):#,pool=None):
    N = RBM.n_v
    M = RBM.n_h
    c = RBM.n_cv
    J = np.zeros([N,N,c,c])
    if pool is None:
        pool = Pool()

    inputs = []
    if subset is not None:
        for i,j in subset:
            inputs.append(i*N+j)
    else:
        for i in range(N):
            for j in range(i,N):
                inputs.append( i*N + j)

    partial_Ker = partial(_Ker_weights_to_couplings_exact,RBM=RBM,data=data,weights=weights,nbins=nbins)
    # res = pool.map(partial_Ker,inputs)
    res = list(map(partial_Ker,inputs) ) # Incompatibility between multiprocessing.Pool and numba. Cannot use parallel.
    for x in range( len(inputs) ):
        i = inputs[x]//N
        j = inputs[x]%N
        J[i,j] = res[x]
        if j !=i:
            J[j,i] = res[x].T
    pool.close()
    return J


def _Ker_weights_to_couplings_exact(x,RBM,data,weights=None,nbins=10):
    N = RBM.n_v
    M = RBM.n_h
    c = RBM.n_cv
    Jij = np.zeros([c,c])
    i = x//N
    j = x%N
    L = layer.Layer(N=1,nature=RBM.hidden)
    tmpW = RBM.weights.copy()
    subsetW = tmpW[:,[i,j],:].copy()
    tmpW[:,[i,j],:] *= 0
    psi_restr = RBM.vlayer.compute_output(data,tmpW)
    for m in range(M):
        count,hist = np.histogram(psi_restr[:,m],bins=nbins,weights=weights)
        hist = (hist[:-1]+hist[1:])/2
        hist_mod = (hist[:,np.newaxis,np.newaxis] + subsetW[m,0][np.newaxis,:,np.newaxis] + subsetW[m,1][np.newaxis,np.newaxis,:]).reshape([nbins*c**2,1])
        for param_name in RBM.hlayer.list_params:
            L.__dict__[param_name][0] = RBM.hlayer.__dict__[param_name][m]
        Phi = utilities.average(L.logpartition(hist_mod).reshape([nbins,c,c]),weights=count)
        Jij += (Phi[:,:,np.newaxis,np.newaxis] + Phi[np.newaxis,np.newaxis,:,:] - Phi[np.newaxis,:,:,np.newaxis].T - Phi[:,np.newaxis,np.newaxis,:]).sum(-1).sum(-1)/c**2
    return Jij
