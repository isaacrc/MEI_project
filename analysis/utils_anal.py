# purpose: resample images in quick function

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

# paths #
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



#### LOAD FUNCTIONS #### 
def load_epi_data(fmri_prep, sub, run, space):
    """
    purpose: load subjects epi data
    inputs:
        - fmri_prep: path to bids epi data
        - sub: subject number
        - run: which run to load
        - space: MNI or subject?
    returns:
        - 4d epi image
    """
    sub_path = os.path.join(fmri_prep, sub, 'ses-01', 'func')
    if space == "MNI":
        epi_in = os.path.join(sub_path, f'%s_ses-01_task-None_run-{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz' % (sub))
    elif space == "T1":
        epi_in = os.path.join(sub_path, f'%s_ses-01_task-None_run-{run:02d}_space-T1w_desc-preproc_bold.nii.gz' % (sub))
    else:
        print("wrong load epi input. check this function")
    epi_data = nib.load(epi_in)
    print("Loading data from %s" % (epi_in))
    print(f'shape of run {run} is {epi_data.shape} \n')

    return epi_data


def load_conf_data(conf_dir, sub, run):
    """
    purpose: load subjects conf data from directory for each run
    inputs:
        - conf_dir: path to conf
        - sub: subject number
        - run: which run to load
    returns:
        - confound file, with array, for that run
    """
    print(f"run: %s_ses-01_task-None_run-{run:02d}_desc-model_timeseries.csv" % (sub))
    conf = pd.read_csv(os.path.join(conf_dir, f"%s_ses-01_task-None_run-{run:02d}_desc-model_timeseries.csv" % (sub)))

    return conf.to_numpy()

#### LOAD FUNCTIONS #### 
def load_epi_sub032(fmri_prep, sub, run, space):
    """
    purpose: load subjects epi data
    inputs:
        - fmri_prep: path to bids epi data
        - sub: subject number
        - run: which run to load
        - space: MNI or subject?
    returns:
        - 4d epi image
    """
    
    if run < 2:
        sub_path = os.path.join(fmri_prep, sub, 'ses-01', 'func')
        epi_in = os.path.join(sub_path, f'%s_ses-01_task-None_run-{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz' % (sub))
    else: 
        ## subtract one run
        sub_path = os.path.join(fmri_prep, sub, 'ses-02', 'func')
        run_adjust = run - 1
        epi_in = os.path.join(sub_path, f'%s_ses-02_task-None_run-{run_adjust:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz' % (sub))
    epi_data = nib.load(epi_in)
    print("Loading data from %s" % (epi_in))
    print(f'shape of run {run} is {epi_data.shape} \n')

    return epi_data

def load_conf_sub032(conf_dir, sub, run):
    """
    purpose: load subjects conf data from directory for each run
    inputs:
        - conf_dir: path to conf
        - sub: subject number
        - run: which run to load
    returns:
        - confound file, with array, for that run
    """
    # session 1 vs session 2 
    print(f"run: %s_ses-01_task-None_run-{run:02d}_desc-model_timeseries.csv" % (sub))
    if run < 2:
        conf = pd.read_csv(os.path.join(conf_dir, f"%s_ses-01_task-None_run-{run:02d}_desc-model_timeseries.csv" % (sub)))
    else:
        conf = pd.read_csv(os.path.join(conf_dir, f"%s_ses-02_task-None_run-{run:02d}_desc-model_timeseries.csv" % (sub)))

    return conf.to_numpy()



def intersect_mask(fmri_prep, sub, num_runs, morph):
    # This is based off of 'load_data' function in template
    # Loads all fMRI runs into a matrix #
    """
    purpose: get an epi mask for each
    fmri_prep: 
    morph = T1 or MNI registration?
    norm_type = by Space or by Time? 
    """
    yoz = []
    print("Begin intersecting, you sexy beast")
    ## preprocess 7 runs ## 
    if sub == 'sub-003' or sub == 'sub-015':
        num_runs+=1
    print(num_runs)
    for run in range(1, num_runs + 1):
        if sub == "sub-012":
            if run >=6:
                run = run+1
        # Load epi data 
        epi = load_epi_data(fmri_prep, sub,run,morph)
        # Mask data
        wholeb_mask = compute_epi_mask(epi) # -- whole brain
        
        yoz.append(wholeb_mask)
    #print(concatenated_data)
    epi_data = nil.masking.intersect_masks(yoz)
    print("all done wit da intersextion (lol)")

    return epi_data


def resample_atlas(atlas_filename, fmri_prep):
    """
    purpose: resample yeo to MEI data
    input: 
    - atlas filename
    - location of preprocessed data for resampling
    output: 
    - atlas_img: the 3d brain image in numpy 2d
    - atlas_nii: the nifti image
    """
    # Load  sample data for resampling
    resamp_run = load_epi_data(fmri_prep, 'sub-007', 2, "MNI")
    # Load parcellation
    d = nib.load(atlas_filename)
    atlas_nii = resample_to_img(d, resamp_run, interpolation='nearest')
    # Get parcellation fdata
    atlas_img = atlas_nii.get_fdata()
    # paracellations scheme
    print(f'count parc:{len(np.unique(atlas_nii.get_fdata()))}')
    print("shape of atlas nii object", atlas_img.shape)
    return atlas_nii, atlas_img

def get_network_labels(num_parc, num_net):
    """
    purpose: get the networks labels from filename
    input: number of parcels, number of networks
    output: networks, network labels and network indices
    """
    ## get filename ##
    label_fn = f'brainiak-aperture-isc-data/Schaefer2018_{num_parc}Parcels_{num_net}Networks_order.txt'
    
    with open(label_fn) as f:
        networks = [' '.join((label.split('_')[1][0], label.split('_')[2]))
                    for label in f.readlines()]

    # Get sorted unique network labels
    idxs = np.unique(networks, return_index=True)[1]
    network_labels = [networks[idx] for idx in sorted(idxs)]

    # Get middle index for each network for plotting -- only necessary for connectivity stuff
    network_idxs = [int(np.median([i for i, n in enumerate(networks)
                                    if n == network]))
                    for network in network_labels]
    print(f'two networks: {network_labels[:2]} \n total nets: {len(network_labels)}')
    
    return networks, network_labels, network_idxs