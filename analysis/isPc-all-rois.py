#!/usr/bin/env python
# coding: utf-8

# # Run ISPC

# This script runs ISPC

# ## py conversion

# In[18]:


#!jupyter nbconvert --to python ISPC.ipynb


# ## Imports 

# In[2]:


import warnings
import sys  
import random
# import logging

import deepdish as dd
import numpy as np

import brainiak.eventseg.event
import nibabel as nib
import nilearn as nil
# Import a function from BrainIAK to simulate fMRI data
import brainiak.utils.fmrisim as sim  

from nilearn.input_data import NiftiMasker

import scipy.io
from scipy import stats
from scipy.stats import norm, zscore, pearsonr
from scipy.signal import gaussian, convolve

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import seaborn as sns 



from brainiak import image, io
from scipy.stats import stats
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

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

import numpy as np 
import os
import os.path
import scipy.io
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask, compute_brain_mask, unmask
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit
from copy import deepcopy

# Brainiak # 
from brainiak import image, io 
from brainiak.isc import (isc, isfc, bootstrap_isc, permutation_isc,
                          timeshift_isc, phaseshift_isc,
                          compute_summary_statistic)
from brainiak.io import load_boolean_mask, load_images
from statsmodels.stats.multitest import multipletests
from nilearn.plotting import plot_stat_map


# In[3]:


random.seed(10)


# ## custom helper functions 

# In[4]:


from utils_anal import load_epi_data, resample_atlas, get_network_labels


# ## directories 


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


## ALL SUBLIST
sub_list = [
    'sub-002', 'sub-003', 'sub-004', 'sub-005','sub-006','sub-007','sub-008','sub-009','sub-010', 'sub-012',
    'sub-013','sub-014', 'sub-015', 'sub-016','sub-017','sub-018','sub-019','sub-020','sub-021',
    'sub-022','sub-023','sub-024','sub-025','sub-026','sub-027','sub-028', 'sub-029', 'sub-030','sub-031','sub-032',
    'sub-033','sub-034','sub-035','sub-036','sub-037','sub-038','sub-039','sub-040', 'sub-041'
]
###### LOADING VARS #######


## TR length of each movie ## 
mov_len_dic = {
'oragami' :  82,
'shrek' : 90,
'sherlock' : 98,
'brushing' : 88,
'cake' : 99,
'office' : 102    
}

voxel_num = 112179

#######################
### VARIABLES ## 
#######################
num_parc = 200 # CHANGE ME

num_net= 7 # CHANGE ME

# which movie repetitions
start_rep = 1
end_rep = 4 # CHANGE ME 

## Static vars ### 

## how many TRs of buffer on the end? ## 
tr_buffer = 4 

# cuttoff the countdown -**effectivly shifts the timeseries over by 4
trim_start = 4 # cuts off first 4 TRs 

## number of runs to iterate over
epi_runs = 6

## load the runs to be included for each subject ## 
sub_run_inc = np.load(behav_dir + '/sub_run_inc.npy', allow_pickle = True).item()

## how many TRs to iterate around the button press
tr_range = np.arange(-3,10)

# this is the window which gannot be impeded by a second button press 
range_len = len(tr_range)
no_interfere = len(np.where(tr_range>0)[0])
print(f'total TRs to be extracted around the bpress: {range_len}\n total trs extracted post bpress: {no_interfere}')

## Load mask
mask_img = nib.load(mask_dir + "/whole_b_bnk.nii.gz")
#mask_img = nib.load(mask_dir + "/shaef_gm_MNI_mask.nii")

# LOAD ATLAS #
## fetch dataset
dataset = datasets.fetch_atlas_schaefer_2018(n_rois=num_parc, yeo_networks = num_net)

# get nii dataset location
atlas_filename = dataset.maps
## get *ROI* atlas labels
labels = dataset.labels

# resample loaded atlas 
atlas_nii, atlas_img = resample_atlas(atlas_filename, fmri_prep)


# ### get network labels

# Load in network labels for each parcell, parcel UNspecific network labels, and the middle parcel within each network
networks, network_labels, network_idxs = get_network_labels(num_parc, num_net)

## create dictionary for all roi, all conditions, all runs ## 
roi_dict = {}

for net_lab in network_labels:
    print(f'start {net_lab}')
    
    ##### Get parcels associated with the target network #### 
    targ_net = (np.array(networks) == net_lab).nonzero()[0] + 1

    ### Get the number of voxels for the target ROI ## 
    roi_tem = np.zeros(atlas_nii.shape)
    
    # set all cases where parcel is equal to roi_num, equal to one, everything else zero (creates a mask)
    for parcel in targ_net:
        roi_tem[atlas_img == parcel] = 1

    # Create a nift image of the mask
    roi_img = nib.Nifti1Image(roi_tem, affine = atlas_nii.affine, header = atlas_nii.header)
    #nib.save(roi_img, rois_dir+'/'+ roi_name.decode("utf-8"))

    # Get labels for parcels in left DMN A network
    vox_roi = len(np.where(roi_tem ==1)[2])
    print(f'using {net_lab} with {vox_roi} voxels')


   
    ## create dictionaries fore each repetition
    m_rep_int = {}
    m_rep_ext = {}
    

    for run in range(start_rep, end_rep + 1):
        # create external and internal dictionaries # 
        external = {}
        internal = {}
        
        for sub in sub_list:
            ### fMRI load ###
            sub_dic_fmri = np.load(f'{preproc_dir}/{sub}_fwhm6_conf.npy', allow_pickle=True).item()
            print(f'start {sub}')
            ## BEHAVIORAL ##
            sub_dic_behav = np.load(os.path.join(behav_dir, f'{sub}_behav.npy'), allow_pickle=True).item()

            # Create subject number 
            sub_num = int(sub[-3:])

            for epi_index in range(0, epi_runs):
                # Add one to the index to create 1-6 runs
                epi_run = epi_index + 1

                # check if run is to be included 
                if not sub_run_inc[sub][epi_run]: continue

                # Get the movie name
                mov_name = sub_dic_behav['mov_order'][epi_index]

                # Create an empty array for the movie runs, append four TRs to account for the 4 trailing TRs, subtract
                # the quantity of TRs that we are trimming from the front 
                mov_runs = np.zeros((range_len, vox_roi, 0))

                print(f'movie: {mov_name} with shape {mov_runs.shape}')

                # Get the fMRI run for the current epi_index
                fmri_run = sub_dic_fmri[epi_run]

                # Loop over runs
                #for run in range(1, m_reps + 1):
                # Is this an internal or external run?
                key = 'External' if (sub_num % 2 == 1 and epi_index < 3) or (sub_num % 2 == 0 and epi_index >= 3) else 'Internal'

                ### only do internal for now ##
                #if key != targ_cond: continue

                ## get behavioral data
                bpress_arr = sub_dic_behav[key][mov_name][f'run-{str(run)}']['bpress']

                ## continue if no button presses
                if bpress_arr == -1:
                    print(f'NO button presses {sub} {run}\n')
                    continue


                # Begin slicing fMRI data #
                start_tr = sub_dic_behav[key][mov_name][f'run-{run:d}']['start_tr']
                end_tr = sub_dic_behav[key][mov_name][f'run-{run:d}']['end_tr']
                run_slice = fmri_run[(start_tr + trim_start):end_tr, :]
                print(f'run{run} bpress count is: {len(bpress_arr)}')
                print(f'start tr {start_tr}, end TR {end_tr}, length of fMRI run {fmri_run.shape}')
                assert fmri_run.shape[0] >= end_tr, 'end TR is greater than fMRI TRs available'

                # calculate differences between button presses, and append the time stamp of the end of scan
                difs = np.diff(np.hstack((bpress_arr, (end_tr - start_tr) * 1.5)))

                ## empty array to append if TR doesn't exist or conflicts with another button pres!
                empty = np.zeros((run_slice.shape[1])) ## number of voxels

                bpress_mat = []
                for idx, bpress in enumerate(bpress_arr):
                    # Find the tr that each onset occured - convert from seconds to TR
                    bpress_tr = round(bpress/1.5)
                    print(f'\ntime of bpress {bpress},  tr is: {bpress_tr}, and next tr is {difs[idx]} away')
                    assert bpress_tr < end_tr, print('TR extends past fMRI data, what on earth is going on!')

                    temp = []
                    ## for each tr in the TR range [likely this is 12 total trs]
                    for num_tr in tr_range:
                        try:
                            # IF tr exists in the range of TRs add to temp array before averaging in the next step
                            current_bpress_tr = run_slice[bpress_tr+num_tr]
                            # if no error is thrown, run a check
                            if difs[idx] < 1.5 * num_tr:
                                temp.append(empty)
                                #print(f'INTERFERE DIF: cur TR is:{num_tr}, next bpress is {int(difs[idx] /1.5)} trs away')
                            else:
                                # add the correct voxel activation for the given TR 
                                temp.append(current_bpress_tr)
                                #print(f'CORRECT {current_bpress_tr[:3]}, length of TR activ is {len(current_bpress_tr)}')

                        except:
                            # if it doesn't exist, add an empty vector
                            temp.append(empty)
                            #print(f'INTERFERE Out of BNDS: bpress_tr {bpress_tr}, index {bpress_tr+num_tr}; total TRs {end_tr - start_tr}')
                        #print(f'FINISHED the {num_tr} TR')
                    bpress_mat.append(temp)
                    #print('FINISHED ONE BUTTN PRESS\n\n')

                ## convert list into numpy array
                bpress_mat = np.dstack(bpress_mat)
                print(f'num bpress: {len(bpress_arr)}, TRs: {bpress_mat.shape[0]}, voxels: {bpress_mat.shape[1]}')
                print(f'shape of bpress for {sub} is {bpress_mat.shape}')
                ## check that we have activation for at least one button press
                assert np.any(bpress_mat), f'NO BPRESSES FOR {sub}'

                ## once we've finished iterating over all button press, average each bpress matrix element-wise!
                #bpress_mat_av = np.mean(bpress_mat, axis =2)
                bpress_mat_av = np.nanmean(np.where(bpress_mat != 0, bpress_mat, np.nan), axis =2)
                # re-insert zeros after averaging, ignoring TRs that had no activation
                bpress_mat_av = np.where(bpress_mat_av != np.nan, bpress_mat_av, 0)

                # convert back to whole brain 4D image, instead of 2d
                bpress_nii = unmask(bpress_mat_av, mask_img)
                print(f'convert bpress averages back to nii: {bpress_nii.shape}')

                # Convert to 4d numpy array
                f_dat_4d = bpress_nii.get_fdata()

                targ_net = (np.array(networks) == net_lab).nonzero()[0] + 1

                # loop through all TRs and get the target voxel pattern #
                bpress_pat = np.column_stack([f_dat_4d[atlas_img == parcel, :].T
                                for parcel in targ_net])

                print(f'now extract voxel TR activations for target ROI: {np.dstack(bpress_pat).shape}\n\n')

                # expand to three dimensions for stacking 
                #bpress_pat_exp = np.expand_dims(np.dstack(bpress_pat), 2)
                bpress_pat_exp = np.expand_dims(bpress_pat, 2)
                #print(f'expand! {bpress_pat.shape}')

                # Stack the run slice with the mov_runs array
                #mov_runs = np.dstack((mov_runs, bpress_pat))
                mov_runs = np.dstack((mov_runs, bpress_pat_exp))
                print(f'stacked! {mov_runs.shape}')
                #assert mov_runs.shape[2] == 4, 'wrong numer of repetitions'

                # set outer loop #
                if key == 'External':
                    target_dict = external
                else:
                    target_dict = internal

                if mov_name not in target_dict:
                    target_dict[mov_name] = mov_runs
                else:
                    #mov_runs = np.expand_dims(mov_runs, 3)
                    target_dict[mov_name] = np.dstack((target_dict[mov_name], mov_runs))
            print(f'\n subject {sub} finished \n')

        ## save into a repetition dictionary ## 
        m_rep_ext[run] = external
        m_rep_int[run] = internal
        print(f'finish {run}')
        
    roi_dict[f'{net_lab}-external'] = m_rep_ext
    roi_dict[f'{net_lab}-internal'] = m_rep_int
    
    ### save ##
    print('saving...')
    np.save(f'{isc_dir}/roi_bpress_ispc.npy', roi_dict)
    print('saving complete ')
    
    print(f'{net_lab } is done')
    
        
            


              
       
# save ##
#print('saving...')
#np.savez_compressed(f'{isc_dir}/ext_isc.npz', **ext_isc)
#np.savez_compressed(f'{isc_dir}/int_isc.npz', **int_isc)
#print('saving complete ')

# In[ ]:





