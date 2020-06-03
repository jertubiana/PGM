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



import sys
sys.path.append('../source/')
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import pandas as pd
import Proteins_utils
import MNIST_utils
import moi
import utilities
import importlib
import pickle

curr_int = np.int16
curr_float = np.float32
encoding = 'latin1'

def load_LatticeProteins(local_env,path=''):
    all_data = Proteins_utils.load_FASTA(path+'Lattice_Proteins_MSA.fasta')
    B = all_data.shape[0]
    seed = utilities.check_random_state(0)
    shuffle = np.argsort(seed.rand(B))
    train_data = all_data[shuffle][:int(0.8*B)]
    test_data = all_data[shuffle][int(0.8*B):]
    train_weights = None
    test_weights = None
    true_contacts_A = [(21,12),(26,3),(9,6),(18,1),(27,8),(25,18),(27,16),
                     (20,1),(19,12),(13,10),(15,8),(19,14),(7,4),
                    (17,14),(25,16),(26,7),(23,2),(23,20),(24,19),(24,15),(15,10),(24,7),(24,11),
                    (23,4),(22,11),(22,5),(25,2),(11,6)]

    contact_map = np.zeros([27,27])
    for i,j in true_contacts_A:
        contact_map[i-1,j-1] = 1
        contact_map[j-1,i-1] = 1

    local_env['train_data'] = train_data.astype(curr_int)
    local_env['test_data'] = test_data.astype(curr_int)
    local_env['train_weights'] = train_weights
    local_env['test_weights'] = test_weights
    local_env['contact_map'] = contact_map.astype(curr_int)




def load_WW(local_env,path=''):
    all_data = Proteins_utils.load_FASTA(path+'WW_domain_MSA.fasta')
    num_neighbours= Proteins_utils.count_neighbours(all_data)
    all_weights = 1.0/num_neighbours
    B = all_data.shape[0]
    seed = utilities.check_random_state(0)
    shuffle = np.argsort(seed.rand(B))
    train_data = all_data[shuffle][:int(0.8*B)]
    test_data = all_data[shuffle][int(0.8*B):]
    train_weights = all_weights[shuffle][:int(0.8*B)]
    test_weights = all_weights[shuffle][int(0.8*B):]
    env = pickle.load(open(path+'WW_test_sequences.data','rb'),encoding=encoding)
    experimental_data = np.asarray(np.concatenate([env['sequences_1'],env['sequences_2'],env['sequences_3'],env['sequences_4']],axis=0),dtype='int')
    experimental_labels = np.asarray(np.concatenate( [np.zeros(len(env['sequences_1'])), np.ones(len(env['sequences_2'])+len(env['sequences_3'])), 2*np.ones(len(env['sequences_4']))],axis=0),dtype='int')

    local_env['train_data'] = train_data.astype(curr_int)
    local_env['test_data'] = test_data.astype(curr_int)
    local_env['train_weights'] = train_weights.astype(curr_float)
    local_env['test_weights'] = test_weights.astype(curr_float)
    local_env['experimental_data'] = experimental_data.astype(curr_int)
    local_env['experimental_labels'] = experimental_labels.astype(curr_int)


def load_Kunitz(local_env,path=''):
    all_data = Proteins_utils.load_FASTA(path+'Kunitz_domain_MSA.fasta')
    num_neighbours= Proteins_utils.count_neighbours(all_data)
    all_weights = 1.0/num_neighbours
    B = all_data.shape[0]
    seed = utilities.check_random_state(0)
    shuffle = np.argsort(seed.rand(B))
    train_data = all_data[shuffle][:int(0.8*B)]
    test_data = all_data[shuffle][int(0.8*B):]
    train_weights = all_weights[shuffle][:int(0.8*B)]
    test_weights = all_weights[shuffle][int(0.8*B):]
    from scipy.io import loadmat
    contact_map = loadmat(path+'contact_map14_extended.mat')['cm'] > 0
    contact_map += contact_map.T # Load contact map.

    local_env['train_data'] = train_data.astype(curr_int)
    local_env['test_data'] = test_data.astype(curr_int)
    local_env['train_weights'] = train_weights.astype(curr_float)
    local_env['test_weights'] = test_weights.astype(curr_float)
    local_env['contact_map'] = contact_map


def load_Hsp70(local_env,path=''):
    all_data, all_labels = Proteins_utils.load_FASTA(path+'Hsp70_protein_MSA.fasta', with_labels = True)
    all_weights = pickle.load(open(path+'Hsp70_info.data','rb'),encoding=encoding)['all_weights']
    B = all_data.shape[0]
    seed = utilities.check_random_state(0)
    shuffle = np.argsort(seed.rand(B))
    train_data = all_data[shuffle][:int(0.8*B)]
    test_data = all_data[shuffle][int(0.8*B):]
    train_weights = all_weights[shuffle][:int(0.8*B)]
    test_weights = all_weights[shuffle][int(0.8*B):]
    local_env['train_data'] = train_data.astype(curr_int)
    local_env['test_data'] = test_data.astype(curr_int)
    local_env['train_weights'] = train_weights.astype(curr_float)
    local_env['test_weights'] = test_weights.astype(curr_float)


def load_Audition_souris(local_env, path=''):
    from scipy.io import loadmat
    all_data = np.asarray(loadmat(path + 'Audition.mat')['binNinf_double_seg'],dtype='int')
    B = all_data.shape[0]
    seed = utilities.check_random_state(0)
    shuffle = np.argsort(seed.rand(B))
    train_data = all_data[shuffle][:int(0.8*B)]
    test_data = all_data[shuffle][int(0.8*B):]
    train_weights = None
    test_weights = None
    local_env['train_data'] = train_data.astype(curr_int)
    local_env['test_data'] = test_data.astype(curr_int)
    local_env['train_weights'] = train_weights
    local_env['test_weights'] = test_weights

def load_Oscillateur_zebrafish(local_env, path=''):
    from scipy.io import loadmat
    all_data = np.asarray(loadmat(path + 'Oscillateur_zebrafish.mat' )['A'],dtype='int')
    B = all_data.shape[0]
    # seed = utilities.check_random_state(0)
    # shuffle = np.argsort(seed.rand(B))
    train_data = all_data[:int(0.8*B)]
    test_data = all_data[int(0.8*B):]
    train_weights = None
    test_weights = None
    local_env['train_data'] = train_data.astype(curr_int)
    local_env['test_data'] = test_data.astype(curr_int)
    local_env['train_weights'] = train_weights
    local_env['test_weights'] = test_weights



def load_MNIST(local_env,path = '../data/MNIST/'):
    train_data = path+u'train-images-idx3-ubyte'
    test_data = path+u't10k-images-idx3-ubyte'
    train_labels = path+u'train-labels-idx1-ubyte'
    test_labels = path+u't10k-labels-idx1-ubyte'

    with open(train_data,'rb') as f:
        data_mnist = MNIST_utils.parse_idx(f)
    with open(train_labels,'rb') as f:
        labels_mnist = MNIST_utils.parse_idx(f)
    with open(test_data,'rb') as f:
        data_mnist_test = MNIST_utils.parse_idx(f)
    with open(test_labels,'rb') as f:
        labels_mnist_test = MNIST_utils.parse_idx(f)

    data_mnist = np.array(data_mnist).reshape([60000,28**2])#.reshape([60000, 28,28])[:,4:-4,4:-4].reshape([60000,400])
    data_mnist_test = np.array(data_mnist_test).reshape([10000,28**2])#.reshape([10000, 28,28])[:,4:-4,4:-4].reshape([10000,400])
    data_mnist = np.array(list(map(lambda x: x>= 128,data_mnist)))
    data_mnist_test = np.array(list(map(lambda x: x>= 128,data_mnist_test)))
    labels_mnist = np.array(labels_mnist)
    labels_mnist_test = np.array(labels_mnist_test)
    train_weights = None
    test_weights = None
    local_env['train_data'] = data_mnist.astype(curr_int)
    local_env['test_data'] = data_mnist_test.astype(curr_int)
    local_env['train_labels'] = labels_mnist.astype(curr_int)
    local_env['test_labels'] = labels_mnist_test.astype(curr_int)
    local_env['train_weights'] = train_weights
    local_env['test_weights'] = test_weights


def load_Silhouettes(local_env,path='../data/Caltech_Silhouettes/'):
    tmp = loadmat(path+'caltech101_silhouettes_28_split1.mat')
    train_weights = None
    test_weights = None
    train_data = np.asarray(tmp['train_data'],dtype='bool')
    test_data = np.asarray(tmp['test_data'],dtype='bool')
    train_labels = np.asarray(tmp['train_labels'],dtype='int')[:,0]
    test_labels = np.asarray(tmp['test_labels'],dtype='int')[:,0]
    local_env['train_data'] = train_data.astype(curr_int)
    local_env['test_data'] = test_data.astype(curr_int)
    local_env['train_labels'] = train_labels.astype(curr_int)
    local_env['test_labels'] = test_labels.astype(curr_int)
    local_env['train_weights'] = train_weights
    local_env['test_weights'] = test_weights


def load_mixture_model(local_env,path = '',N=1000,M=5,n_c=1,nature='Spin'):
    try:
        env = pickle.load(open(path+ 'mixture_model_N_%s_M_%s_nc_%s_nature_%s.data'%(N,M,n_c,nature) ,'rb'))
    except:
        MoI = moi.MoI(N=N,M=M,nature=nature,n_c=n_c)
        if nature == 'Bernoulli':
            MoI.cond_muv = 0.99*(np.random.randint(0,high=2,size=[MoI.M,MoI.N]).astype(np.float32)-1)/2 + 0.5
        elif nature == 'Spin':
            MoI.cond_muv = 0.99*(2*np.random.randint(0,high=2,size=[MoI.M,MoI.N]).astype(np.float32)-1)
        elif nature == 'Potts':
            dominant = np.random.randint(0,high=n_c, size=[MoI.M,MoI.N])
            MoI.cond_muv = 0.01 * np.ones([MoI.M,MoI.N,MoI.n_c],dtype=np.float32)/(n_c-1)
            for m in range(MoI.M):
                for n in range(MoI.N):
                    MoI.cond_muv[m,n, dominant[m,n]] = 0.99
        train_data,_ = MoI.gen_data(10000)
        test_data,_ = MoI.gen_data(10000)
        train_data = np.asarray(train_data,dtype=np.int16)
        test_data = np.asarray(test_data,dtype=np.int16)
        train_weights = None
        test_weights = None
        env = {'train_data':train_data,'test_data':test_data,'train_weights':train_weights,
        'test_weights':test_weights,'MoI':MoI}
        pickle.dump( env, open(path+ 'mixture_model_N_%s_M_%s_nc_%s_nature_%s.data'%(N,M,n_c,nature) ,'wb') )

    local_env['train_data'] = env['train_data']
    local_env['test_data'] = env['test_data']
    local_env['train_weights'] = env['train_weights']
    local_env['test_weights'] = env['test_weights']
    local_env['MoI'] = env['MoI']



def infer_type_data(data):
    N = data.shape[1]
    mini = data.min()
    maxi = data.max()
    if mini == -1:
        nature = 'Spin'
        n_c = 1
    else:
        if maxi == 1:
            nature = 'Bernoulli'
            n_c = 1
        else:
            nature = 'Potts'
            n_c = maxi +1
    return nature,N,n_c
