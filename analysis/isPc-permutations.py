#!/usr/bin/env python
# coding: utf-8

# # runs permutations on isPC/FC matrices

# see above lol -- copied from isPC-vis




# ## Imports 

# In[121]:


import warnings
import sys  
import random
import os
import os.path

import deepdish as dd
import numpy as np
import pandas as pd

import scipy.io
from scipy import stats
from scipy.stats import stats
from scipy.stats import norm, zscore, pearsonr
from scipy.signal import gaussian, convolve

#plotting
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import seaborn as sns 

# nil and nib 
import nibabel as nib
import nilearn as nil

from nilearn.input_data import NiftiMasker
from nilearn import datasets, plotting
from nilearn.plotting import plot_roi
from nilearn.input_data import NiftiSpheresMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import concat_imgs, resample_img, mean_img,index_img
from nilearn import image
from nilearn import masking
from nilearn.plotting import view_img
from nilearn.image import resample_to_img
from nilearn.image import concat_imgs, resample_img, mean_img
from nilearn.plotting import view_img
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask, compute_brain_mask, unmask
from nilearn.plotting import plot_stat_map

# Brainiak # 
from brainiak import image, io 
import brainiak.utils.fmrisim as sim  
from brainiak import image, io
import brainiak.eventseg.event
from brainiak.isc import (isc, isfc, bootstrap_isc, permutation_isc,
                          timeshift_isc, phaseshift_isc,
                          compute_summary_statistic)
from brainiak.io import load_boolean_mask, load_images
from statsmodels.stats.multitest import multipletests

from brainiak.isc import squareform_isfc


# In[ ]:





# In[122]:


random.seed(10)


# ## custom helper functions 

# In[123]:


from utils_anal import load_epi_data, resample_atlas, get_network_labels


# ## directories 

# In[124]:


top_dir = '/jukebox/graziano/coolCatIsaac/MEI'
data_dir = top_dir + "/data"
work_dir = data_dir + '/work'
mask_dir = work_dir + '/masks'
behav_dir = top_dir + '/data/behavioral'
rois_dir = data_dir + "/rois"
fmri_prep = data_dir + '/bids/derivatives/fmriprep'
conf_dir = work_dir + '/confs'
preproc_dir = work_dir + '/preproc'
isc_dir = work_dir + '/isc_dat'
perm_dir = work_dir + '/perms'


# # FUNCTIONS

def perm_matrices(targ_dic, roi, num_perms):
    """
    purpose: get permutations across all matrices for an ROI
    input: target dictionary and the roi for analysis
    return: array of permuted accuracies
    """
    num_subs = targ_dic[roi].shape[2]
    perm_arr = []

    for perm in range(num_perms):
        # Transpose input data to compute intersubject pattern correlation
        ispcs = isfc(np.rollaxis(targ_dic[roi], 1, 0),
                            pairwise=False, vectorize_isfcs=False)

        ## create random array of 1s and 0s to randomly sign flip each matrix
        rand_arr = np.random.choice([-1, 1], size=num_subs, replace=True)
        # add dimensions to rand_arr so that it can be broadcast to each matrix
        result = rand_arr[:, np.newaxis, np.newaxis] * ispcs
        # get the average across the 18, 13, 13 matrix -- results should be 13 x 13
        av_across_subs = np.nanmean(result, axis = 0)
        # Convert these directly to condensed ISFCs (and ISCs)
        ispcs_c, iscs = squareform_isfc(av_across_subs)
        print(f"Condensed ISFCs shape: {ispcs_c.shape}, "
              f"ISCs shape: {iscs.shape}")
        ## get shape for transformation later
        iscs_shape = iscs.shape[0]
        perm_arr.append(np.hstack((ispcs_c, iscs)))
    perm_arr = np.vstack(perm_arr) 
    return perm_arr, iscs_shape


# ### load data
#roi_bpress_ispc = np.load(f'{isc_dir}/roi_bpress_ispc.npy', allow_pickle = True).item()
roi_bpress_ispc = np.load(f'{isc_dir}/targ_net_roi_bpress_ispc.npy', allow_pickle = True).item()
print(f'total anals: {len(list(roi_bpress_ispc.keys()))}')


# VARIABLES **changeme**
targ_cond = "internal"
roi = 'LH_Default_PFC_11'
targ_run = 1
num_perm = 1000
pval = .05

# DICS N SHLISTS
targ_dic = {} 
sav_dic = {}
perm_dic = {}
mov_list = ['sherlock', 'office', 'oragami', 'shrek', 'brushing', 'cake']
# ASSEMBLE TARGET DATA
for roi in 
for targ_mov in mov_list:
    targ_keys = [key for key in list(roi_bpress_ispc.keys()) if key[-8:] == targ_cond]  
    #targ_keys = [key for key in targ_keys if key[:1] == hemi]  
    for key in targ_keys:
        targ_dic[key[:-9]] = roi_bpress_ispc[key][targ_run][targ_mov]


    print(f'total ROIS: {len(list(targ_dic.keys()))}  \n {targ_mov} {targ_run} \n {targ_cond}')
    print(f' subjects included: {targ_dic[list(targ_dic.keys())[0]].shape[2]}') 





    # ### start 
    # get permutated accuracies 
    perm_mat, isc_shape = perm_matrices(targ_dic, roi, num_perm)
    # get the maximum of each permutation test. This controls across all timepoints -- most stringent.
    # should use this for
    max_perm = np.max(perm_mat, axis=1)
    ## get the off diagnonal for plotting [not really important]
    perm_mat_c = np.max(perm_mat, axis=0)
    ## save perm_mat for fun
    np.save(f'{perm_dir}/{targ_mov}_{targ_cond}_{roi}_run{targ_run}_permutations.npy', perm_mat)

    # get actual correlation matrix 
    ispcs = isfc(np.rollaxis(targ_dic[roi], 1, 0),
                        pairwise=False, vectorize_isfcs=False,
                        summary_statistic='mean')
    ## flatten and concatenate
    ispcs_c, iscs = squareform_isfc(ispcs)
    # this is the flattened *observed values
    cat_ispcs = np.hstack((ispcs_c, iscs))


    ## get p-values 
    ## the current val must beat ALL timepoints in ALL simulations 
    sig_vals =  [(np.sum(val < perm_mat)) / perm_mat.shape[0] for val in cat_ispcs]
    sig_val_bin = [1 if val < pval else 0 for val in sig_vals ]    


    # In[206]:


    # Convert these directly back to redundant ISFCs
    off_diag = np.array(sig_val_bin[:-isc_shape])
    diag = np.array(sig_val_bin[-isc_shape:])
    sig_mat_r = squareform_isfc(off_diag, diag)
    print(f"Converted redundant ISFCs shape: {sig_mat_r.shape}")


    # In[223]:


    mats = np.dstack((sig_mat_r, ispcs))
    sav_dic[targ_mov] = mats
    np.save(f'{isc_dir}/{targ_cond}_{roi}_run{targ_run}.npy', sav_dic)




