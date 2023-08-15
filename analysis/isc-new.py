#!/usr/bin/env python
# coding: utf-8

# # Run ISC

# This script runs ISC

# ## py conversion

# In[1]:


#jupyter nbconvert --to python slurm_create-data_preproc.ipynb


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
from nilearn.masking import compute_epi_mask, compute_brain_mask
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

# In[ ]:





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

### sub_list ###
"""

========================================================
===== ~~~ Summary ALL ~~~ ===== 
========================================================
Initially exclude: sub-001, sub-003, sub-011, sub-012, sub-014, sub-015, sub-029 (3+ runs still available tho)
Maybe exclude: sub-014, sub-024, sub-037 (No button presses)
Maybe exclude: sub-020, sub-022, sub-031
total exclude = 10


~~~~~~~ BEHAVIORAL ~~~~~~~~~~
~~~ Summary behavioral ~~~

========================================================
========================================================
No button presses: sub-014, sub-024, sub-037 

========================================================
========================================================
sub-003: Extra row observed for external, which was manually deleted. Should be fine to rerun -- i removed from bad_subs
    - current brushing: 87, TRUE: 88 for THREE runs. last run when there was an issue and scan had to be restarted
    - Scan 6 == 1 run of brushing
    - Scan 7 = 3 runs of brushing 
    - ** no idea when run 7 was started tho! will prolly need to throw out! For now we can process all
sub-014: NO internal OR external bpress, curious! ---should be fixed now
    - changed excel data to include two null columns
    - ALSO only 30 TRs for the second run epi data. i went into the room to adjust runny eyes. Data for this run is not usable. but after should
      be okay if u want to put in the effort 
sub-015: less than 24 runs ---should be fixed now w custom code (but keep out of main run cuz will throw an error otherwise )
    - SUMMARY: need to exclude the office run 3 for fMRI but can include run 3 in behavioral, no data at all for CAKE
    - DID NOT add the 'mov_name' component to the custom code, will need to implment from current iteration
    - **** Use external, ignore internal for now... cuz internal is fuqd. Can preprocess all 7 tho

~~~~~~~ FMRI ~~~~~~~~~~
~~~ Summary fMRI ~~~
sub002: 
 - appended two TRs onto the last run, copied from the third to last TR. should be good to use
sub-003, sub-012, sub-014, sub-015, sub-029: 
 - Five runs usable, will need to incorporate them at some point 
sub-001: external usable - first three runs 

========================================================
========================================================
subject 002: (usable, with adjustment)
    only has 92 INSTEAD OF 94 TRs for the FINAL run of shrek. i think cuz i turned off the scanner
    too soon, which didn't account for 4 TRs of buffer? Yep! end tr is 390, (388, 112179). so duplicate TRs maybe.
    - Temporarily eliminate! or duplicate TRs 
    
sub-003: (usable, 5 runs)
- Scan 6 == 1 run of brushing
- Scan 7 = 3 runs of brushing 
*** need to re-preprocess, then append 6 and 7 together; or just throw out this one cuz idk when scan started

sub-011: (usable)
- not preprocessed 
sub-012 (5 runs usable, one run idk)
- I'll need to post process. It seems that an earbud fell out while reading the directions for run 5 
summary: the fifth run  needs to be discarded. all others are usable.
see the behav data: 
sub_dic['External']['oragami']
{'run-1': {'start_tr': 94, 'end_tr': 188, 'bpress': -1},
 'run-2': {'start_tr': 188, 'end_tr': 282, 'bpress': [100.81657150003593]},
 'run-3': {'start_tr': 282, 'end_tr': 376, 'bpress': [88.26579949003644]},
 'run-4': {'start_tr': 378,
  'end_tr': 472,
  }
- you can see taht the start TR is 94!! that's becuz i had to go into the scan room during the instructions screen.
- this is producing the wrong indexing -- becuz the scan doesn't contain enough TRs according to the behavioral data:
    - run 4 of shrek only has 12 TRs because we start at TR 94.
- im not sure when the scan actually started, but if i wanted to try to include i could 
    set the start TR to 1 -- essentially suggesting that the first TR was collected on the external 'waiting 
    for TR' flip. might as well try at some point, but exclude for now
    - this would involve just subtracting 94 from all start end end TRs i think
See the excel behavioral file for further notes notes
this 

sub-014 - no button press data and 7 runs (5 runs usable, but no bpress)
    - same problem as above -- i went into the scanner room during the instructions period to wipe eyes
    {'run-1': {'start_tr': 237, 'end_tr': 323, 'bpress': -1},
     'run-2': {'start_tr': 329, 'end_tr': 415, 'bpress': -1},
     'run-3': {'start_tr': 417, 'end_tr': 503, 'bpress': -1},
     'run-4': {'start_tr': 509, 'end_tr': 595, 'bpress': -1}}

sub-029: (5 runs usable)
- scanner malfunction, maybe possible to stitch together run 6 cuz i started on the *tenth* TR

========================================================
========================================================
Left handed: (sub-020, sub-022, sub-032)
    - sub-020: also no button presses, appeared to be awake tho
    - sub-022: lefty
    - sub-031: ambidexterious, great data
========================================================
========================================================
"""

## ALL SUBLIST
sub_list = [
    'sub-002', 'sub-003', 'sub-004', 'sub-005','sub-006','sub-007','sub-008','sub-009','sub-010', 'sub-012',
    'sub-013','sub-014', 'sub-015', 'sub-016','sub-017','sub-018','sub-019','sub-020','sub-021',
    'sub-022','sub-023','sub-024','sub-025','sub-026','sub-027','sub-028', 'sub-029', 'sub-030','sub-031','sub-032',
    'sub-033','sub-034','sub-035','sub-036','sub-037','sub-038','sub-039','sub-040', 'sub-041'
]



## imports
from utils_anal import load_epi_data, resample_atlas, get_network_labels


# ## custom helper functions 
def reshape_dictionary(data, num_repeats = 4):
    """
    purpose: create a 4d array sorted into x runs per subject
    input: dictionary of stacked subject data 
    output: 4d array sorted into runs
    """
    output_dict = {}
    for mov in data:
        print(f'{mov} size is {data[mov].shape}')
        dim_3 = num_repeats
        dim_4 = int(data[mov].shape[2] / num_repeats)
        
        if key not in output_dict:
            output_dict[mov] = data[mov].reshape(*data[mov].shape[:-1], dim_3, dim_4, order = 'F')
        
        #print(np.array_equal(data[mov][:, :, 1], output_dict[mov][:, :, 0, 1]))
        #print(np.array_equal(data[mov][:, :, 17], output_dict[mov][:, :, 1, 0]))
        print(np.array_equal(data[mov][:, :, 4], output_dict[mov][:, :, 0, 1]))
        print(f'reshaped {output_dict[mov].shape}')
    print()
    return output_dict



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


##### Variables for preproc ### 

## load the runs to be included for each subject ## 
sub_run_inc = np.load(behav_dir + '/sub_run_inc.npy', allow_pickle = True).item()

## how many TRs on the end? ## 
tr_buffer = 4 

## how many TRs to trim from front? 
tr_trim = 4 

## how man EPI runs
epi_runs = 6
start_rep = 1 
end_rep = 4


##### Variables for ISC ### 
n_boot = 1000 # how many boostraps?
p_thresh = .05

### static vars ### 
run_list = [1, 2, 3, 4]
cond_list = ["internal", "external"]
mov_list = ['office', 'brushing', 'oragami', 'shrek', 'cake', 'sherlock']
thresh_vis_dic = {cond: {mov: {run: {} for run in run_list} for mov in mov_list} for cond in cond_list}
nothresh_vis_dic = {cond: {mov: {run: {} for run in run_list} for mov in mov_list} for cond in cond_list}


## mask info ##
ref_nii = nib.load(mask_dir + "/whole_b_bnk.nii.gz")
mask_img = load_boolean_mask(mask_dir + "/whole_b_bnk.nii.gz")
mask_coords = np.where(mask_img)



######## BEGIN ########
# set empty dic
external = {}
internal = {}

for sub in sub_list:
    ### fMRI load ###
    sub_dic_fmri = np.load(f'{preproc_dir}/{sub}_fwhm6_conf.npy', allow_pickle=True).item()
    print(f'start {sub}')
    ## BEHAVIORAL ##
    sub_dic_behav = np.load(os.path.join(behav_dir, f'{sub}_behav.npy'), allow_pickle=True).item()

    # Initialize an empty dictionary to store the stacked arrays for the current subject
    stacked_arrays = {}
    
    # Create subject number 
    sub_num = int(sub[-3:])

    for epi_index in range(0, epi_runs):
        # Add one to the index to create 1-6 runs
        epi_run = epi_index + 1
        
        # check if run is to be included 
        if not sub_run_inc[sub][epi_run]: continue

        # Get the movie name
        mov_name = sub_dic_behav['mov_order'][epi_index]

        # Create an empty array for the movie runs
        mov_runs = np.zeros((mov_len_dic[mov_name] + tr_buffer - tr_trim, voxel_num, 0))

        print(f'movie: {mov_name} with shape {mov_runs.shape}')

        # Get the fMRI run for the current epi_index
        fmri_run = sub_dic_fmri[epi_run]
        
        # Loop over runs
        for run in range(start_rep, end_rep + 1):
            # Is this an internal or external run?
            key = 'External' if (sub_num % 2 == 1 and epi_index < 3) or (sub_num % 2 == 0 and epi_index >= 3) else 'Internal'
            

            # Begin slicing #
            start_tr = sub_dic_behav[key][mov_name][f'run-{run:d}']['start_tr']
            end_tr = sub_dic_behav[key][mov_name][f'run-{run:d}']['end_tr']
            run_slice = fmri_run[(start_tr + tr_trim):end_tr, :]
            
            # BEHAV checks #
            assert mov_runs.shape[0] == mov_len_dic[mov_name] + tr_buffer - tr_trim, f'behavioral ERROR: movie: {mov_name}, epi {epi_run}, run: {run}'

            # fMRI data check #
            assert run_slice.shape[0] == mov_runs.shape[0] + tr_buffer - tr_trim, f'fMRI error! {run_slice.shape[0]} verse {mov_runs.shape[0]}'
        
            # Stack the run slice with the mov_runs array
            mov_runs = np.dstack((mov_runs, run_slice))
            print(f'stacked! {mov_runs.shape}')

        
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

            
print('done')
                 
## Reshape Dictionary
ext_isc = reshape_dictionary(external)
int_isc = reshape_dictionary(internal)

## save if you'd like 
np.savez_compressed(f'{isc_dir}/ext_isc.npz', **ext_isc)
np.savez_compressed(f'{isc_dir}/int_isc.npz', **int_isc)

## load if you'd like 
#ext_isc = np.load(f'{isc_dir}/ext_isc.npz')
#int_isc = np.load(f'{isc_dir}/int_isc.npz')


################### ISC ANAL ################## #

for cond in cond_list:
    print(f'START {cond}')
    if cond == "external":
        targ_dic = ext_isc
    else:
        targ_dic = int_isc
    for run in run_list:
        for mov in mov_list:
    
            # select movie and run from loaded data ** account for indexing
            data = targ_dic[mov][...,run - 1,:]
            print(f'{mov} run {run} shape is: {data.shape}')

            # Z-score time series for each voxel
            data = zscore(data, axis=0)

            # Leave-one-out approach
            iscs = isc(data, pairwise=False, tolerate_nans=.8)

            # Check shape of output ISC values
            print(f"ISC values shape = {iscs.shape} \ni.e., {iscs.shape[0]} "
                  f"left-out subjects and {iscs.shape[1]} voxel(s)")

            # Compute mean ISC (with Fisher transformation)
            mean_iscs = compute_summary_statistic(iscs, summary_statistic='mean', axis=0)

            print(f"ISC values shape = {mean_iscs.shape} \ni.e., {mean_iscs.shape[0]} "
                  f"mean value across left-out subjects and {iscs.shape[1]} voxel(s)"
                  f"\nMinimum mean ISC across voxels = {np.nanmin(mean_iscs):.3f}; "
                  f"maximum mean ISC across voxels = {np.nanmax(mean_iscs):.3f}")


            # Compute median ISC
            median_iscs = compute_summary_statistic(iscs, summary_statistic='median',
                                                    axis=0)

            print(f"ISC values shape = {median_iscs.shape} \ni.e., {median_iscs.shape[0]} "
                  f"median value across left-out subjects and {iscs.shape[1]} voxel(s)"
                  f"\nMinimum median ISC across voxels = {np.nanmin(median_iscs):.3f}; "
                  f"maximum median ISC across voxels = {np.nanmax(median_iscs):.3f}")

            # Run bootstrap hypothesis test on ISCs
            observed, ci, p, distribution = bootstrap_isc(iscs, pairwise=False,
                                                          ci_percentile=95,
                                                          summary_statistic='median',
                                                          n_bootstraps=n_boot)


            # Get number of NaN voxels
            n_nans = np.sum(np.isnan(observed))
            print(f"{n_nans} voxels out of {observed.shape[0]} are NaNs "
                  f"({n_nans / observed.shape[0] * 100:.2f}%)")

            # Get voxels without NaNs
            nonnan_mask = ~np.isnan(observed)
            nonnan_coords = np.where(nonnan_mask)

            # Mask both the ISC and p-value map to exclude NaNs
            nonnan_isc = observed[nonnan_mask]
            nonnan_p = p[nonnan_mask]

            # Get FDR-controlled q-values
            nonnan_q = multipletests(nonnan_p, method='fdr_by')[1]
            print(f"{np.sum(nonnan_q < p_thresh)} significant voxels "
                  f"controlling FDR at {p_thresh}")
            
            # non thresholded iscs
            nothresh_nonnan_isc = nonnan_isc.copy()
            # Threshold ISCs according FDR-controlled threshold
            nonnan_isc[nonnan_q >= p_thresh] = np.nan

            # Reinsert thresholded ISCs back into whole brain image
            isc_thresh = np.full(observed.shape, np.nan)
            isc_thresh[nonnan_coords] = nonnan_isc
            
            # Reinsert NON-thresholded ISCs back into whole brain image
            isc_nt = np.full(observed.shape, np.nan)
            isc_nt[nonnan_coords] = nothresh_nonnan_isc

            # Create empty 3D image and populate
            # with thresholded ISC values
            isc_img = np.full(ref_nii.shape, np.nan)
            isc_img[mask_coords] = isc_thresh
            
            # Create empty 3D image and populate
            # with NON-thresholded ISC values
            nt_isc_img = np.full(ref_nii.shape, np.nan)
            nt_isc_img[mask_coords] = isc_nt

            # Convert to NIfTI image
            isc_nii = nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header)
            isc_nii_nt = nib.Nifti1Image(nt_isc_img, ref_nii.affine, ref_nii.header)
            
            # Save into dictionary
            thresh_vis_dic[cond][mov][run] = isc_nii
            nothresh_vis_dic[cond][mov][run] = isc_nii_nt
            print(f'\nfinish {cond}-{mov}-{run}\n')
            
            
            
            # DO rois # -- this is now completed in the visual script



                    
## save visualization for sig voxels
np.save(f'{isc_dir}/isc_sigvox_nii_dic.npy', thresh_vis_dic)
## save visualization for no threshold
np.save(f'{isc_dir}/isc_noThresh_nii_dic.npy', nothresh_vis_dic)
## save correlation dictionary
#np.save(f'{isc_dir}/{out_name}', cor_dic) -- see visual script 


