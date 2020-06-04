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

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import sys,os

import numpy as np
import Proteins_utils
try:
    import Proteins_3D_utils
except:
    print('Missing packages for importing Proteins_3D_utils')
import RBM_utils
import sequence_logo
import matplotlib.image as mpimg



curr_int = np.int16
curr_float = np.float32

def get_ax(ax,i,nrows,ncols):
    if (ncols>1) & (nrows>1):
        col = i%ncols
        row = i//ncols
        ax_ = ax[row,col]
    elif (ncols>1) & (nrows==1):
        ax_ = ax[i]
    elif (ncols==1) & (nrows>1):
        ax_ = ax[i]
    else:
        ax_ = ax
    return ax_

def clean_ax(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])



def plot_input_mean(RBM,I, subset, I_range=None,mean=None,weights = None, ax = None, ncols = 1, xlabels = None, with_dots = None , figsize = (3,3), show= True):
    if type(subset) in [int,np.int64]:
        subset = [subset]
    if ax is not None:
        nfeatures = 1
        ncols = 1
        nrows = 1
        return_fig = False
    else:
        nfeatures = len(subset)
        nrows = int(np.ceil(nfeatures/float(ncols)))
        fig, ax = plt.subplots(nrows,ncols)
        fig.set_figheight( nrows * figsize[0] )
        fig.set_figwidth( ncols * figsize[1] )
        return_fig = True

    if I_range is None:
        I_min = I.min()
        I_max = I.max()
        I_range = (I_max-I_min) * np.arange(0,1+0.01,0.01) + I_min
    I_range = I_range.astype(curr_float)

    if mean is None:
        mean = RBM.hlayer.mean_from_inputs(np.repeat(I_range[:,np.newaxis],RBM.n_h,axis=1))

    if xlabels is None:
        xlabels = [r'Input $I_{%s}$'%(i+1) for i in range(nfeatures)]

    for i in range(nfeatures):
        ax_ = get_ax(ax,i,nrows,ncols)
        ax2_ = ax_.twinx()
        ax2_.hist(I[:,subset[i]],density=True,weights=weights,bins=100)
        ax_.plot(I_range,mean[:,subset[i]],c='black',linewidth=2)

        xmin = I[:,subset[i]].min()
        xmax = I[:,subset[i]].max()
        ymin = mean[:,subset[i]].min()
        ymax = mean[:,subset[i]].max()



        if with_dots is not None:
            for val in with_dots[i]:
                tmp= np.argmin(np.abs(mean[:,subset[i]]-val) )
                ax_.plot(I_range[tmp],mean[tmp,subset[i]],'.',c='red',markersize=20)
                xmin = min(I_range[tmp],xmin)
                xmax = max(I_range[tmp],xmax)
                ymin = min(mean[tmp,subset[i]], ymin  )
                ymax = max(mean[tmp,subset[i]], ymax  )


        ax_.set_xlim([xmin,xmax])
        step = int( (xmax-xmin )/4.0) +1
        xticks = np.arange(int(xmin), int(xmax)+1, step)
        ax_.set_xticks(xticks)
        ax_.set_xticklabels(xticks,fontsize=12)
        ax_.set_ylim([ymin,ymax])
        step = int( (ymax-ymin )/4.0)+1
        yticks = np.arange(int(ymin), int(ymax)+1, step)
        ax_.set_yticks(yticks)
        ax_.set_yticklabels(yticks,fontsize=12)
        ax2_.set_yticks([])
        for tl in ax_.get_yticklabels():
            tl.set_fontsize(14)
        ax_.set_zorder(ax2_.get_zorder()+1)
        ax_.patch.set_visible(False)

        ax_.set_xlabel(xlabels[i],fontsize=14)

    for i in range(nfeatures,ncols*nrows):
        ax_ = get_ax(ax,i,nrows,ncols)
        clean_ax(ax_)

    if return_fig:
        plt.tight_layout()
        if show:
            plt.show()
        return fig



def plot_input_classes(subset,I_background, I_test, class_names=None, background_name = 'MSA',
                colors = None, xlabels = None , markers = None, beta_order=False,show=True,fontsize=10,nbins=10,ncols=1,ymax=None):
    n_class = len(I_test)
    if class_names is None:
        class_names = ['Class %s'%i for i in range(n_class)]
    if colors is None:
        colors = ['C%s'%(i%10) for i in range(n_class)]

    if type(nbins)==int:
        nbins = [nbins for _ in range(n_class)]

    nfeatures = len(subset)
    nrows = int(np.ceil(nfeatures/float(ncols)))
    fig, ax = plt.subplots(nrows, ncols)
    fig.set_figheight(nrows * 3)
    fig.set_figwidth(ncols * 3)

    all_labels = [background_name] + class_names
    if xlabels is None:
        xlabels = [r'$I_{%s}$'%(i+1) for i in range(nfeatures)]

    for i in range(nfeatures):
        ax_ = get_ax(ax,i,nrows,ncols)
        ax2_ = ax_.twinx()
        row = i/ncols
        col = i%ncols

        all_hists = []
        ax_.hist(I_background[:,subset[i]],color='black',bins=100,density=True, histtype='bar',alpha=0.25,label=background_name)
        F = patches.Rectangle((0,0),1,1,facecolor='black',alpha=0.25)
        all_hists.append(F)
        for k in range(n_class):
            ax2_.hist(I_test[k][:,subset[i]],color=colors[k],density=True, histtype='step',label=class_names[k],bins=nbins[k])
            F = patches.Rectangle((0,0),1,1,facecolor=colors[k])
            all_hists.append(F)
        ax_.set_xlabel(xlabels[i],fontsize=14)
        ax_.set_yticks([])
        ax2_.set_yticks([])
        if ymax is not None:
            ax_.set_ylim([0,ymax[i]])
        if row==0:
            ax_.legend(all_hists[ (col*(n_class+1))//ncols : ((col+1)*(n_class+1))//ncols ] , all_labels[ (col*(n_class+1))//ncols : ((col+1)*(n_class+1))//ncols ] ,fontsize=fontsize,frameon=False,ncol=1)
        for line in all_hists:
            line.set_visible(False)
    for i in range(nfeatures,ncols*nrows):
        ax_ = get_ax(ax,i,nrows,ncols)
        clean_ax(ax_)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_input_classes_scatter(subsets,I_background, I_classes, class_names=None,
                  background_name = 'MSA',figsize=8,class_colors = None, axis_labels = None ,
                  markers = None,show=True,background_markersize=0.5,markersize=3,fontsize=10,ncols=1,ncols_legend=1,
                 xlim = None,ylim=None):

    n_class = len(I_classes)

    if class_names is None:
        class_names = ['Class %s'%i for i in range(n_class)]
    if class_colors is None:
        class_colors = ['C%s'%(i%10) for i in range(n_class)]

    if markers is None:
        markers = ['o' for i in range(n_class)]


    try:
        x = subsets[0][0]
    except:
        subsets = [subsets]

    nfeatures = len(subsets)
    nrows = int(np.ceil(nfeatures/float(ncols)))
    fig, ax = plt.subplots(nrows, ncols)
    if type(figsize) == list:
        fig.set_figwidth(ncols * figsize[0])
        fig.set_figheight(nrows * figsize[1])
    else:
        fig.set_figheight(nrows * figsize)
        fig.set_figwidth(ncols * figsize)

    all_labels = [background_name] + class_names

    if axis_labels is None:
        axis_labels = [ [r'$I_{%s}$'%(subset[0]+1), r'$I_{%s}$'%(subset[1]+1) ] for subset in subsets]

    if xlim is None:
        xlim = [None for l in range(nfeatures)]
    if ylim is None:
        ylim = [None for l in range(nfeatures)]

    for l in range(nfeatures):
        ax_ = get_ax(ax,l,nrows,ncols)
        row = l/ncols
        col = l%ncols
        i = subsets[l][0]
        j = subsets[l][1]

        all_scats = []
        S = ax_.scatter(I_background[:,i],I_background[:,j] ,c='gray',s=background_markersize,label=background_name,alpha=0.25)
        all_scats.append(S)
        for k in range(n_class):
            S=ax_.scatter(I_classes[k][:,i],I_classes[k][:,j],c=class_colors[k],s=markersize,label=class_names[k],marker=markers[k])
            all_scats.append(S)

        ax_.set_xlabel(axis_labels[l][0],fontsize=fontsize)
        ax_.set_ylabel(axis_labels[l][1],fontsize=fontsize)

        if row==0:
            ax_.legend(all_scats[ (col*(n_class+1))//ncols : ((col+1)*(n_class+1))//ncols ] , all_labels[ (col*(n_class+1))//ncols : ((col+1)*(n_class+1))//ncols ] ,fontsize=16,frameon=False,ncol=ncols_legend,markerscale=2.0,handletextpad=0.2,loc='best')

        if xlim[l] is None:
            mini = I_background[:,i].min()
            maxi = I_background[:,i].max()
            xlim[l]  =  [mini- 0.25 * (maxi-mini), maxi+ 0.25 * (maxi-mini)]
        if ylim[l] is None:
            mini = I_background[:,j].min()
            maxi = I_background[:,j].max()
            ylim[l]  =  [mini- 0.25 * (maxi-mini), maxi+ 0.25 * (maxi-mini)]



        ax_.set_xlim(xlim[l])
        ax_.set_ylim(ylim[l])


    for i in range(nfeatures,ncols*nrows):
        ax_ = get_ax(ax,i,nrows,ncols)
        clean_ax(ax_)

    plt.tight_layout()
    if show:
        plt.show()
    return fig




def plot_top_activating_distance(RBM, I,data, subset, nseqs = 20,all_distances = None, distance_top_features = None,ax=None,xlabels=None,ncols=1,figsize=(3,3),show=True):
    if type(subset) in [int,np.int64]:
        subset = [subset]
    if all_distances is None:
        sub = np.argsort(np.random.randn(len(data)))[:1000]
        all_distances = Proteins_utils.distance(data[sub]).flatten()

    if distance_top_features is None:
        I_ = I.copy()
        if RBM.hidden == 'dReLU':
            eta = RBM.hlayer.eta
            I_ *= np.sign(eta)[np.newaxis,:]
        q = np.percentile(I_,100* (1-float(nseqs)/data.shape[0]),axis=0)
        distance_top_features = []
        for k in range(RBM.n_h):
            subgroup = data[I_[:,k]>q[k]]
            distance_top_features.append( Proteins_utils.distance(subgroup).flatten() )

    if ax is not None:
        nfeatures = 1
        ncols = 1
        nrows = 1
        return_fig = False
    else:
        nfeatures = len(subset)
        nrows = int(np.ceil(nfeatures/float(ncols)))
        fig, ax = plt.subplots(nrows,ncols)
        fig.set_figheight( nrows * figsize[0] )
        fig.set_figwidth( ncols * figsize[1] )
        return_fig = True

    if xlabels is None:
        xlabels = ['Distance %s'%(i+1) for i in range(nfeatures)]

    for i in range(nfeatures):
        ax_ = get_ax(ax,i,nrows,ncols)
        ax_.hist(all_distances,bins=RBM.n_v,density=True,range=(0,1),color = 'gray',alpha = 0.5)
        ax_.hist(distance_top_features[subset[i]],bins=RBM.n_v,density=True,range=(0,1),alpha=0.5)
        ax_.set_xticks([0,0.5,1])
        ax_.set_xticklabels([0,0.5,1],fontsize=12)
        ax_.set_yticks([])
        ax_.set_xlabel(xlabels[i],fontsize=14)
    for i in range(nfeatures,ncols*nrows):
        ax_ = get_ax(ax,i,nrows,ncols)
        clean_ax(ax_)

    if return_fig:
        plt.tight_layout()
        if show:
            plt.show()
        return fig


def scatter_distance_fitness( Dmins, Fs, colors = None, names = [], ylabel = None,
                            non_linearity = None, nmax = 200, s1 =14, s2=16, s3=2,s4 =12,figsize=(5,5),
                             off = 0.15, w = 0.15,xticks = [0,5,10,15], yticks = None, show_points=None):

    n_samples = len(Dmins)
    if colors is None:
        colors = ['C%s'%k for k in range(n_samples)]
    if names is None:
        names = ['Method %s'%(k+1) for k in range(n_samples)]
    if ylabel is None:
        ylabel = 'Likelihood'

    if show_points is None:
        show_points = [True for k in range(n_samples)]

    if non_linearity is None:
        non_linearity = lambda x: x
    nFs = [non_linearity(F) for F in Fs]

    offs = off * (np.arange(n_samples) - (n_samples-1)/2.)


    fig , ax = plt.subplots()
    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])
    fig.subplots_adjust(left=0.2,bottom=0.15)

    subset_synths = [np.argsort(np.random.randn(len(nF)))[:nmax] for nF in nFs]


    for l in range(n_samples):
        if show_points[l]:
            plt.scatter(Dmins[l][subset_synths[l]]+ offs[l], nFs[l][subset_synths[l]],
                    marker='o', s = s3, c = colors[l],label=names[l]);


    for l in range(n_samples):
        mu_dist = Dmins[l].mean()
        mu_nF = nFs[l].mean()
        covariance = np.cov(Dmins[l],  nFs[l] )
        lam,v = np.linalg.eigh(covariance)

        ell = patches.Ellipse(xy=(mu_dist, mu_nF),
                      width=np.sqrt(lam)[0]*2, height=np.sqrt(lam)[1]*2,
                      angle=np.rad2deg(-np.arccos(v[0, 0])))
        ell.set_facecolor(colors[l])
        ell.set_alpha(0.25)
        ax.add_artist(ell)


    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks,fontsize=s1);
    ax.set_xlabel('Mutations to closest natural sequence',fontsize=s2)
    ax.set_ylabel(ylabel,fontsize=s2)

    if yticks is not None:
        ax.set_yticks( [non_linearity(ytick) for ytick in yticks] )
        ax.set_yticklabels(yticks,fontsize=s1);

    for tl in ax.get_xticklabels():
        tl.set_fontsize(s1)
    for tl in ax.get_yticklabels():
        tl.set_fontsize(s1)
    plt.legend(fontsize=s4,frameon=False,loc='lower left',markerscale=2,handletextpad=-0.3);
    return fig



def make_all_weights(RBM,data, pdb_file = None, pdb_chain=None, subset=None,name = 'all_weights.pdf',
    weights_per_page = 10, rows_per_weight = 1, weights=None, gap_at_bottom = None,
    figsize=None,sort='importance', dpi=None,ticks_every=5,
    visualize_sector_kwargs={},theta_important=0.4):

    if dpi is None:
        if pdb_file is not None:
            dpi = 100
        else:
            dpi = 50

    mini_name = name[:-4]
    n_h = RBM.n_h
    I = RBM.input_hiddens(data)
    I_min = I.min()
    I_max = I.max()
    I_range = np.asarray( (I_max-I_min) * np.arange(0,1+0.01,0.01) + I_min, dtype=curr_float)
    mean = RBM.hlayer.mean_from_inputs(np.repeat(I_range[:,np.newaxis],n_h,axis=1))

    norms = RBM_utils.get_norm(RBM.weights,include_gaps=True)

    if gap_at_bottom is None:
        if RBM.n_cv == 21:
            gap_at_bottom = True
        else:
            gap_at_bottom = False

    if sort == 'norm':
        sorting_value = norms
    elif sort == 'p':
        p = RBM_utils.get_sparsity(RBM.weights,include_gaps=True)
        sorting_value = - p
    elif sort == 'jump':
        jump = RBM_utils.get_hlayer_jump(RBM.weights,positive_only=False)
        sorting_value = jump
    elif sort == 'importance':
        importance = RBM_utils.get_hidden_unit_importance(RBM,data,weights=weights)
        sorting_value = importance
    else:
        sorting_value = np.arange(n_h)

    if gap_at_bottom:
        gap_fraction = RBM_utils.get_norm_gaps(RBM.weights,a=1)/RBM_utils.get_norm(RBM.weights,a=1)
        order = np.argsort(sorting_value  + 1000. * (gap_fraction < 0.2) )[::-1]
    else:
        order = np.argsort(sorting_value)[::-1]

    if subset is not None:
        order = subset

    n_h_shown = len(order)

    if sort == 'importance':
        titles = [r'$\Delta \mathcal{L} = %.3f$'%importance[order[i]] for i in range(n_h_shown)]
    else:
        titles = [r'$||W||_2 = %.2f$'%norms[order[i]] for i in range(n_h_shown)]
    ylabels = ['Weights %s (%s)'%(i+1,order[i]) for i in range(n_h_shown)]

    sub = np.argsort(np.random.randn(len(data)))[:1000]
    all_distances = Proteins_utils.distance(data[sub]).flatten()

    I_ = I.copy()
    if RBM.hidden == 'dReLU':
        eta = RBM.hlayer.eta
        I_ *= np.sign(eta)[np.newaxis,:]

    q = np.percentile(I_,100* (1-20.0/data.shape[0]),axis=0)

    distance_top_features = []
    for k in range(n_h):
        subgroup = data[I_[:,k]>q[k]]
        distance_top_features.append( Proteins_utils.distance(subgroup).flatten() )


    if pdb_file is not None:
        structure_folder = mini_name + '_' + 'images_structures/'
        norm_1 = np.abs(RBM.weights).sum(-1)
        maxi = norm_1.max(-1)
        important = norm_1/maxi[:,np.newaxis] > theta_important
        sectors = [ np.nonzero(important[l])[0] for l in order ]
        sector_names = ['absolute_%s_order_%s'%(order[i],i+1) for i in range(n_h_shown)]

        if not 'npixels' in visualize_sector_kwargs.keys():
            visualize_sector_kwargs['npixels'] = 1000
        if not 'with_numbers' in visualize_sector_kwargs.keys():
            visualize_sector_kwargs['with_numbers'] = True
        if not 'with_numbers_every' in visualize_sector_kwargs.keys():
            visualize_sector_kwargs['with_numbers_every'] = 2
        if not 'pdb_numbers' in visualize_sector_kwargs.keys():
            visualize_sector_kwargs['pdb_numbers'] = False
        if not 'turn' in visualize_sector_kwargs.keys():
            visualize_sector_kwargs['turn'] = None
        if not 'chain' in visualize_sector_kwargs.keys():
            visualize_sector_kwargs['chain'] = None
        if not 'first_model_only' in visualize_sector_kwargs.keys():
            visualize_sector_kwargs['first_model_only'] = True
        if not 'sector_colors' in visualize_sector_kwargs.keys():
            visualize_sector_kwargs['sector_colors'] = None
        if not 'chain_colors' in visualize_sector_kwargs.keys():
            visualize_sector_kwargs['chain_colors'] = None
        if not 'show_sidechains' in visualize_sector_kwargs.keys():
            visualize_sector_kwargs['show_sidechains'] = True

        Proteins_3D_utils.visualize_sectors(sectors, pdb_file, structure_folder,
                              sector_names = sector_names,
                              alignment = data, simultaneous = False,  exit = True,
                              save = True,**visualize_sector_kwargs)

    if figsize is None:
        figsize = (  max(int(0.3 * RBM.n_v/rows_per_weight), 2)  ,  3*rows_per_weight)


    n_pages = int(np.ceil( n_h_shown/float(weights_per_page)))
    for k in range(n_pages):
        if (rows_per_weight>1) & (pdb_file is not None):
            fig = plt.figure(figsize = (figsize[0]+1.5*figsize[1],figsize[1]*weights_per_page))
            gs = gridspec.GridSpec(2*weights_per_page*rows_per_weight, 3,width_ratios = [figsize[0],figsize[1]/2,figsize[1]])
        elif (rows_per_weight==1) & (pdb_file is not None):
            fig = plt.figure(figsize = (figsize[0]+2*figsize[1],figsize[1]*weights_per_page))
            gs = gridspec.GridSpec(2*weights_per_page, 3,width_ratios = [figsize[0],figsize[1],figsize[1]])

        elif (rows_per_weight>1) & (pdb_file is None):
            fig = plt.figure(figsize = (figsize[0]+figsize[1],figsize[1]*weights_per_page))
            gs = gridspec.GridSpec(2*weights_per_page*rows_per_weight, 2,width_ratios = [figsize[0],figsize[1]])
        else:
            fig = plt.figure(figsize = (figsize[0]+figsize[1],figsize[1]*weights_per_page))
            gs = gridspec.GridSpec(2*weights_per_page*rows_per_weight, 2,width_ratios = [figsize[0],figsize[1]])


        for i in range(k*weights_per_page,(k+1)*weights_per_page):
            ii = i%weights_per_page
            if (rows_per_weight>1) & (pdb_file is not None):
                ax1 = [fig.add_subplot(gs[2*rows_per_weight*ii+2*l:2*rows_per_weight*ii+2*(l+1), 0]) for l in range(rows_per_weight)]
                ax2 = fig.add_subplot(gs[2*ii*rows_per_weight:(2*ii+1)*rows_per_weight, 1])
                ax3 = fig.add_subplot(gs[(2*ii+1) * rows_per_weight: 2*(ii+1) * rows_per_weight, 1])
                ax4 = fig.add_subplot(gs[2*ii * rows_per_weight: 2*(ii+1) * rows_per_weight, 2])
            elif (rows_per_weight==1) & (pdb_file is not None):
                ax1 = fig.add_subplot(gs[2*ii:2*ii+2, 0])
                ax2 = fig.add_subplot(gs[2*ii, 1])
                ax3 = fig.add_subplot(gs[2*ii+1, 1])
                ax4 = fig.add_subplot(gs[2*ii:2*ii+2, 2])
            elif (rows_per_weight>1) & (pdb_file is None):
                ax1 = [fig.add_subplot(gs[2*rows_per_weight*ii+2*l:2*rows_per_weight*ii+2*(l+1), 0]) for l in range(rows_per_weight)]
                ax2 = fig.add_subplot(gs[2*ii*rows_per_weight:(2*ii+1)*rows_per_weight, 1])
                ax3 = fig.add_subplot(gs[(2*ii+1) * rows_per_weight: 2*(ii+1) * rows_per_weight, 1])
            else:
                ax1 = fig.add_subplot(gs[2*ii:2*ii+2, 0])
                ax2 = fig.add_subplot(gs[2*ii, 1])
                ax3 = fig.add_subplot(gs[2*ii+1, 1])

            sequence_logo.Sequence_logo(RBM.weights[order[i]],ax=ax1,
                ylabel = ylabels[i], title=titles[i]
                ,ticks_every=ticks_every,ticks_labels_size=14,title_size=20,show=False,nrows=rows_per_weight)

            plot_input_mean(RBM,I, order[i], I_range=I_range,mean=mean,weights = weights, ax = ax2,xlabels=[r'$I_{%s}$'%(i+1)])

            plot_top_activating_distance(RBM, I, None,order[i], nseqs = 20,all_distances = all_distances, distance_top_features = distance_top_features,ax=ax3,xlabels=['Hamming Distance'])
            if pdb_file is not None:
                img = mpimg.imread(structure_folder+'sector_' + sector_names[i] + '.png')
                rows = (img.sum(-1) == 1.*3).min(1) # Remove white
                cols = (img.sum(-1) == 1.*3).min(0)
                ax4.imshow(img[~rows,:][:,~cols])
                ax4.axis('off')

        plt.tight_layout()
        fig.savefig(mini_name+'tmp_#%s.png'%k,dpi=dpi)
        fig.clear()
        plt.close(fig)

    command = 'pdfjoin ' + mini_name+'tmp_#*.png -o %s'%name
    os.system(command)
    command = 'rm '+mini_name+'tmp_#*.png'
    os.system(command)
    print('Make all weights: Done.')
    return 'done'
