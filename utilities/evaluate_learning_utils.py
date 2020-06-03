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
import pandas as pd
import sys
sys.path.append('../source/')
import moi
import utilities
import matplotlib.pyplot as plt
import matplotlib
import importlib
import pickle
from plots_utils import clean_ax
import Proteins_utils
import dataset_utils
from sklearn.linear_model import LogisticRegressionCV

curr_int = np.int16
curr_float = np.float32

def aa_color_scatter(letter):
    if letter == '-':
        return 'yellow'
    else:
        return Proteins_utils.aa_color(letter)



def assess_moment_matching(RBM, data, data_gen,datah_gen=None, weights=None,with_reg=True,show=True):
    h_data = RBM.mean_hiddens(data)
    h_gen = RBM.mean_hiddens(data_gen)
    mu = utilities.average(data,c=RBM.n_cv,weights=weights)
    mu_gen = utilities.average(data_gen,c=RBM.n_cv)

    mu_h = utilities.average(h_data,weights=weights)
    mu_h_gen = h_gen.mean(0)

    if RBM.n_cv>1:
        cov_vh = utilities.average_product(h_data,data,c2=RBM.n_cv,weights=weights) - mu[np.newaxis,:,:] * mu_h[:,np.newaxis,np.newaxis]
    else:
        cov_vh = utilities.average_product(h_data,data,c2=RBM.n_cv,weights=weights) - mu[np.newaxis,:] * mu_h[:,np.newaxis]


    if datah_gen is not None:
        mu_data_gen = RBM.mean_visibles(datah_gen)
        if RBM.n_cv>1:
            cov_vh_gen = utilities.average_product(datah_gen, mu_data_gen, mean2=True,c2=RBM.n_cv) - mu[np.newaxis,:,:] * mu_h[:,np.newaxis,np.newaxis]
        else:
            cov_vh_gen = utilities.average_product(datah_gen, mu_data_gen, mean2=True,c2=RBM.n_cv) - mu[np.newaxis,:] * mu_h[:,np.newaxis]
    else:
        if RBM.n_cv>1:
            cov_vh_gen = utilities.average_product(h_gen,data_gen,c2=RBM.n_cv) - mu_gen[np.newaxis,:,:] * mu_h_gen[:,np.newaxis,np.newaxis]
        else:
            cov_vh_gen = utilities.average_product(h_gen,data_gen,c2=RBM.n_cv) - mu_gen[np.newaxis,:] * mu_h_gen[:,np.newaxis]


    if RBM.hidden == 'dReLU':
        I_data = RBM.vlayer.compute_output(data,RBM.weights)
        I_gen = RBM.vlayer.compute_output(data_gen,RBM.weights)

        mu_p_pos,mu_n_pos,mu2_p_pos,mu2_n_pos = RBM.hlayer.mean12_pm_from_inputs(I_data)
        mu_p_pos = utilities.average(mu_p_pos,weights = weights)
        mu_n_pos = utilities.average(mu_n_pos,weights = weights)
        mu2_p_pos = utilities.average(mu2_p_pos,weights=weights)
        mu2_n_pos = utilities.average(mu2_n_pos,weights=weights)

        mu_p_neg,mu_n_neg,mu2_p_neg,mu2_n_neg = RBM.hlayer.mean12_pm_from_inputs(I_gen)
        mu_p_neg = utilities.average(mu_p_neg)
        mu_n_neg = utilities.average(mu_n_neg)
        mu2_p_neg = utilities.average(mu2_p_neg)
        mu2_n_neg = utilities.average(mu2_n_neg)

        a = RBM.hlayer.gamma
        eta = RBM.hlayer.eta
        theta = RBM.hlayer.delta

        moment_theta = -mu_p_pos/np.sqrt(1+eta) + mu_n_pos/np.sqrt(1-eta)
        moment_theta_gen = -mu_p_neg/np.sqrt(1+eta) + mu_n_neg/np.sqrt(1-eta)
        moment_eta = 0.5 * a/(1+eta)**2 * mu2_p_pos - 0.5 * a/(1-eta)**2 * mu2_n_pos + theta/(2*np.sqrt(1+eta)**3) * mu_p_pos - theta/(2*np.sqrt(1-eta)**3) * mu_n_pos
        moment_eta_gen = 0.5 * a/(1+eta)**2 * mu2_p_neg - 0.5 * a/(1-eta)**2 * mu2_n_neg + theta/(2*np.sqrt(1+eta)**3) * mu_p_neg - theta/(2*np.sqrt(1-eta)**3) * mu_n_neg

        moment_theta *=-1
        moment_theta_gen *=-1
        moment_eta *=-1
        moment_eta_gen *=-1

    W = RBM.weights
    if with_reg:
        l2 = RBM.l2
        l1 = RBM.l1
        l1b = RBM.l1b
        l1c= RBM.l1c
        l1_custom = RBM.l1_custom
        l1b_custom = RBM.l1b_custom
        n_c2 = RBM.n_cv
        if l2>0:
            cov_vh_gen += l2 * W
        if l1>0:
            cov_vh_gen += l1 * np.sign( W)
        if l1b>0: # NOT SUPPORTED FOR POTTS
            if n_c2 > 1: # Potts RBM.
                cov_vh_gen += l1b * np.sign(W) *  np.abs(W).mean(-1).mean(-1)[:,np.newaxis,np.newaxis]
            else:
                cov_vh_gen += l1b * np.sign( W) * (np.abs(W).sum(1))[:,np.newaxis]
        if l1c>0: # NOT SUPPORTED FOR POTTS
            cov_vh_gen += l1c * np.sign( W) * ((np.abs(W).sum(1))**2)[:,np.newaxis]

        if any([l1>0,l1b>0,l1c>0]):
            mask_cov = np.abs(W)>1e-3
        else:
            mask_cov = np.ones(W.shape,dtype='bool')
    else:
        mask_cov = np.ones(W.shape,dtype='bool')

    if RBM.n_cv>1:
        if RBM.n_cv == 21:
            list_aa = Proteins_utils.aa
        else:
            list_aa = Proteins_utils.aa[:-1]
        colors_template = np.array([matplotlib.colors.to_rgba(aa_color_scatter(letter)) for letter in list_aa] )
        color = np.repeat(colors_template[np.newaxis,:,:],data.shape[1],axis=0).reshape([data.shape[1]* RBM.n_cv,4])
    else:
        color = 'C0'

    s2 = 14

    if RBM.hidden == 'dReLU':
        fig, ax = plt.subplots(3,2)
        fig.set_figheight(3*5)
        fig.set_figwidth(2*5)
    else:
        fig, ax = plt.subplots(2,2)
        fig.set_figheight(2*5)
        fig.set_figwidth(2*5)
    clean_ax(ax[1,1])

    ax_ = ax[0,0]
    ax_.scatter(mu.flatten(),mu_gen.flatten(),c=color);
    ax_.plot([mu.min(),mu.max()],[mu.min(),mu.max()])
    ax_.set_xlabel(r'$<v_i>_d$',fontsize=s2);
    ax_.set_ylabel(r'$<v_i>_m$',fontsize=s2);
    r2_mu = np.corrcoef(mu.flatten(),mu_gen.flatten() )[0,1]**2
    error_mu =  np.sqrt(  ( (mu-mu_gen)**2/( mu * (1-mu) + 1e-4 ) ).mean() )
    mini = mu.min()
    maxi = mu.max()
    ax_.text( 0.6 * maxi + 0.4 * mini,0.25 * maxi + 0.75 * mini,r'$R^2 = %.2f$'%r2_mu,fontsize=s2)
    ax_.text( 0.6 * maxi + 0.4 * mini,0.35 * maxi + 0.65 * mini,r'$Err = %.2e$'%error_mu,fontsize=s2)
    ax_.set_title('Mean visibles',fontsize=s2)


    ax_ = ax[0,1]
    ax_.scatter(mu_h,mu_h_gen);
    ax_.plot([mu_h.min(),mu_h.max()],[mu_h.min(),mu_h.max()])
    ax_.set_xlabel(r'$<h_\mu>_d$',fontsize=s2);
    ax_.set_ylabel(r'$<h_\mu>_m$',fontsize=s2);
    r2_muh = np.corrcoef(mu_h,mu_h_gen)[0,1]**2
    error_muh =  np.sqrt( ( (mu_h - mu_h_gen)**2).mean() )
    mini = mu_h.min()
    maxi = mu_h.max()
    ax_.text( 0.6 * maxi + 0.4 * mini,0.25 * maxi + 0.75 * mini,r'$R^2 = %.2f$'%r2_muh,fontsize=s2)
    ax_.text( 0.6 * maxi + 0.4 * mini,0.35 * maxi + 0.65 * mini,r'$Err = %.2e$'%error_muh,fontsize=s2)
    ax_.set_title('Mean hiddens',fontsize=s2)



    ax_ = ax[1,0]
    if RBM.n_cv>1:
        color = np.repeat(np.repeat(colors_template[np.newaxis,np.newaxis,:,:],RBM.n_h,axis=0),data.shape[1] ,axis=1).reshape([RBM.n_v * RBM.n_h * RBM.n_cv,4])
        color = color[mask_cov.flatten()]
    else:
        color = 'C0'

    cov_vh = cov_vh[mask_cov].flatten()
    cov_vh_gen = cov_vh_gen[mask_cov].flatten()

    ax_.scatter(cov_vh,cov_vh_gen,c = color);
    ax_.plot([cov_vh.min(),cov_vh.max()],[cov_vh.min(),cov_vh.max()])
    ax_.set_xlabel(r'Cov$(v_i \;, h_\mu)_d$',fontsize=s2)
    ax_.set_ylabel(r'Cov$(v_i \;, h_\mu)_m + \nabla_{w_{\mu i}} \mathcal{R}$',fontsize=s2)
    r2_vh = np.corrcoef(cov_vh,cov_vh_gen)[0,1]**2
    error_vh =  np.sqrt( ( (cov_vh - cov_vh_gen)**2).mean() )
    mini = cov_vh.min()
    maxi = cov_vh.max()
    ax_.text( 0.6 * maxi + 0.4 * mini,0.25 * maxi + 0.75 * mini,r'$R^2 = %.2f$'%r2_vh,fontsize=s2)
    ax_.text( 0.6 * maxi + 0.4 * mini,0.35 * maxi + 0.65 * mini,r'$Err = %.2e$'%error_vh,fontsize=s2)
    ax_.set_title('Hiddens-Visibles correlations',fontsize=s2)

    if RBM.hidden == 'dReLU':
        ax_ = ax[2,0]
        ax_.scatter(moment_theta,moment_theta_gen,c=theta);
        ax_.plot([moment_theta.min(),moment_theta.max()],[moment_theta.min(),moment_theta.max()])
        ax_.set_xlabel(r'$<-\frac{\partial E}{\partial \theta}>_d$',fontsize=s2);
        ax_.set_ylabel(r'$<-\frac{\partial E}{\partial \theta}>_m$',fontsize=s2);
        r2_theta = np.corrcoef(moment_theta,moment_theta_gen )[0,1]**2
        error_theta =  np.sqrt( ( (moment_theta - moment_theta_gen)**2).mean() )
        mini = moment_theta.min()
        maxi = moment_theta.max()
        ax_.text( 0.6 * maxi + 0.4 * mini,0.25 * maxi + 0.75 * mini,r'$R^2 = %.2f$'%r2_theta,fontsize=s2)
        ax_.text( 0.6 * maxi + 0.4 * mini,0.35 * maxi + 0.65 * mini,r'$Err = %.2e$'%error_theta,fontsize=s2)
        ax_.set_title('Moment theta',fontsize=s2)

        ax_ = ax[2,1]
        ax_.scatter(moment_eta,moment_eta_gen,c=np.abs(eta) );
        ax_.plot([moment_eta.min(),moment_eta.max()],[moment_eta.min(),moment_eta.max()])
        ax_.set_xlabel(r'$<-\frac{\partial E}{\partial \eta}>_d$',fontsize=s2);
        ax_.set_ylabel(r'$<-\frac{\partial E}{\partial \eta}>_m$',fontsize=s2);
        r2_eta = np.corrcoef(moment_eta,moment_eta_gen )[0,1]**2
        error_eta =  np.sqrt( ( (moment_eta - moment_eta_gen)**2).mean() )
        mini = moment_eta.min()
        maxi = moment_eta.max()
        ax_.text( 0.6 * maxi + 0.4 * mini,0.25 * maxi + 0.75 * mini,r'$R^2 = %.2f$'%r2_eta,fontsize=s2)
        ax_.text( 0.6 * maxi + 0.4 * mini,0.35 * maxi + 0.65 * mini,r'$Err = %.2e$'%error_eta,fontsize=s2)
        ax_.set_title('Moment eta',fontsize=s2)

    plt.tight_layout()
    if show:
        fig.show()

    if RBM.hidden == 'dReLU':
        errors = [error_mu,error_muh,error_vh,error_theta,error_eta]
        r2s = [r2_mu,r2_muh,r2_vh,r2_theta,r2_eta]
    else:
        errors = [error_mu,error_muh,error_vh]
        r2s = [r2_mu,r2_muh,r2_vh]

    return fig, errors, r2s


def likelihood(model, data, data_test, weights=None,weights_test=None,n_betas_AIS=20000,M_AIS=10):
    model.AIS(n_betas=n_betas_AIS,M=M_AIS,beta_type='linear')
    l = utilities.average(model.likelihood(data),weights=weights)
    l_test = utilities.average(model.likelihood(data_test),weights=weights_test)
    return [l,l_test]


def auto_correl(data,nmax=None,nature='Bernoulli',n_c=1):
    B = data.shape[0]
    L = data.shape[1]
    N = data.shape[2]
    if nmax is None:
        nmax = L/2

    if n_c ==1:
        data_hat = np.fft.fft(np.real(data),axis=1)
        C = np.real(np.fft.ifft(np.abs(data_hat)**2,axis=1) ).mean(0).mean(-1)/float(L)
        mu = data.mean(0).mean(0)
        if nature == 'Bernoulli':
            C_hat = 1 + 2*C - 2 * mu.mean() - (mu**2+ (1-mu)**2).mean()
        elif nature == 'Spin':
            C_hat = (1+C)/2 - ( ((1+mu)/2)**2 + ((1-mu)/2)**2).mean()
        return C_hat[:nmax]/C_hat[0]
    else:
        C = np.zeros(L)
        mu = utilities.average(data.reshape([B*L,N]),c=n_c)
        for c in range(n_c):
            data_ = (data == c)
            data_hat = np.fft.fft(np.real(data_),axis=1)
            C+= np.real(np.fft.ifft(np.abs(data_hat)**2,axis=1) ).mean(0).mean(-1)/float(L)
        C_hat = C - (mu**2).mean() * n_c
        return C_hat[:nmax]/C_hat[0]


def get_inception_score(data_gen,dataset,weights=None, path_data = 'data/', path_classifiers = 'classifiers/',M=20,eps=1e-10):
    try:
        classifier = pickle.load(open(path_classifiers+'%s_Classifier.data'%dataset,'rb'))['classifier']
    except:
        print('Learning a classifier first....')
        train_env = {}
        exec('dataset_utils.load_%s(train_env,path=path_data)'%dataset)
        if 'train_labels' in train_env.keys():
            classifier = LogisticRegressionCV(n_jobs=5,multi_class='multinomial')
            classifier.fit(train_env['train_data'],train_env['train_labels'])
        else:
            nature, N, n_c = dataset_utils.infer_type_data(train_env['train_data'])
            classifier = moi.MoI(nature=nature, N = N,M=M,n_c=n_c)
            classifier.fit(train_env['train_data'],verbose=0,weights=train_env['train_weights'])
        pickle.dump({'classifier':classifier},open(path_classifiers+'%s_Classifier.data'%dataset,'wb'))
    if hasattr(classifier,'predict_proba'):
        probas = classifier.predict_proba(data_gen)
    elif hasattr(classifier,'expectation'):
        probas = classifier.expectation(data_gen)
    else:
        print('No expectation or predict_proba from classifier')
        return
    proba_av = utilities.average(probas, weights=weights)
    scores = (probas * np.log( (probas+eps)/(proba_av+eps) ) ).sum(-1)
    inception_score = np.exp(utilities.average(scores,weights=weights) )
    return inception_score
