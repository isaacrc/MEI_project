#!/usr/bin/env python3

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend such as Agg
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from scipy.stats import zscore
from nilearn.plotting import view_img, plot_stat_map
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from himalaya.kernel_ridge import KernelRidgeCV
from voxelwise_tutorials.voxelwise_tutorials.delayer import Delayer 
from sklearn.pipeline import make_pipeline
from himalaya.scoring import correlation_score
from himalaya.kernel_ridge import Kernelizer, ColumnKernelizer
from himalaya.kernel_ridge import MultipleKernelRidgeCV

# Define the directory to save results
results_dir = 'model1/model_1_results_banded_ridge/stacked'

# Check if the directory exists; if not, create it
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
## Define model

# Ridge regression with alpha grid and nested CV
alphas = np.logspace(1, 10, 10)
inner_cv = KFold(n_splits=4)
outer_cv = LeaveOneOut()
loo = LeaveOneOut()

# Mean-center each feature (columns of predictor matrix)
scaler = StandardScaler(with_mean=True, with_std=True)

# Create delays at 3, 4.5, 6 seconds (1.5 s TR)
delayer = Delayer(delays=[2, 3, 4])

# Define Model
#ridge = KernelRidgeCV(alphas=alphas, cv=inner_cv) ## inner cv!
#ridge = KernelRidgeCV(alphas=alphas)

# Ridge regression with alpha grid and nested CV
solver = 'random_search'
n_iter = 20
solver_params = dict(n_iter=n_iter, alphas=alphas)

# Banded ridge regression with column kernelizer
banded_ridge = MultipleKernelRidgeCV(kernels="precomputed", solver=solver,
                                     solver_params=solver_params, cv=inner_cv)


# Filenames for intact notthefall data and Schaefer atlas
func_fn = ('sub-284_task-black_space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz')
atlas_fn = ('Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2.5mm.nii.gz')

# Load in the Schaefer 400-parcel atlas
atlas_nii = nib.load(atlas_fn)
atlas_img = atlas_nii.get_fdata()

# Define a function to load GloVe embeddings from a file
def load_glove_embeddings(file_path):
    with open(file_path) as f:
        glove = json.load(f)
    return glove

# Define the list of movies, runs, and data types
movies = ["shrek", "brushing"]
runs = range(1, 5)  # Runs 1 to 4
data_types = ['int', 'ext']

# Dictionary containing the number of TRs for each movie
mov_len_dic = {
    'oragami': 82,
    'shrek': 90,
    'sherlock': 98,
    'brushing': 88,
    'cake': 99,
    'office': 102
}

# Define the directory where the files are located
data_dir = 'model1/model_1_fmri_data'

    
# Loop through movies
for mov in movies:
    # Load the specific GloVe embeddings for the current movie
    glove_file = f'embeddings/{mov}_embeddings.json'
    glove = load_glove_embeddings(glove_file)

    # Load in time-stamped transcript
    transcript_fn = f'embeddings/{mov}_transcript_final.csv'
    transcript = pd.read_csv(transcript_fn, sep=',')

    # Stimulus properties based on the dictionary
    tr = 1.5
    stim_dur = mov_len_dic[mov] * tr
    stim_trs = mov_len_dic[mov]

    # Convert transcript to list for simplicity
    transcript = transcript.values.tolist()

    # Loop through TRs
    transcript_trs = []

    # Loop through TRs
    for t in np.arange(stim_trs):
        # Container for words in this TR
        tr_words = []

        try:
            # Check if upcoming word onset is in this TR
            while t * tr < transcript[0][2] <= t * tr + tr:
                # If so, pop this word out of list and keep it
                w = transcript.pop(0)
                tr_words.append(w[0])
        except:
            transcript_trs.append(tr_words)
            continue

        # Append words and move to the next TR
        transcript_trs.append(tr_words)

    # Load list of standard stop words
    stopwords = np.load('nltk_stopwords.npy')
    append = ["farquad", "there's", "let's", "I", "I'm"]
    stopwords = np.append(stopwords, append)

    # Initialize the predictor matrix
    glove_dim = 300
    embeddings = []

    # Assign GloVe embeddings to each TR:
    for t in transcript_trs:
        embeddings_tr = []

        # Grab the embedding for each word in a TR
        for w in t:
            # Ignore stop words
            if w not in stopwords:
                embeddings_tr.append(np.array(glove.get(w, np.zeros(glove_dim))).astype(float))

        # For non-empty TRs, average the embeddings
        if len(embeddings_tr) > 0:
            embeddings_tr = np.mean(embeddings_tr, axis=0)
        else:
            embeddings_tr = np.zeros(glove_dim)
        embeddings.append(embeddings_tr)

    embeddings = np.stack(embeddings, axis=0)

    # Number of rows populated with zeros
    zero_rows_mask = (np.sum(embeddings, axis=1) == 0)
    # make it numeric
    zero_rows_mask_num = (np.sum(embeddings, axis=1) == 0).astype(int)
    # reshape it to size (TRs, 1)
    zero_rows_mask_num = zero_rows_mask_num.reshape(-1, 1)
    # for the regression
    nuis_regress = zero_rows_mask_num
    # Count the number of rows with all zeros
    num_zero_rows = np.count_nonzero(zero_rows_mask)
    

    print(f"Number of rows with all zeros for {mov}: {num_zero_rows}")

    # Get the indices of rows with all zeros
    zero_rows_indices = np.where(zero_rows_mask)[0]

    print(f"Indices of rows with all zeros for {mov}:")
    print(zero_rows_indices)

    # Initialize a list to store pairs of matching row indices
    matching_rows = []

    # Iterate through all pairs of rows in the embeddings array
    for i in range(embeddings.shape[0]):
        for j in range(i + 1, embeddings.shape[0]):
            if np.array_equal(embeddings[i], embeddings[j]):
                matching_rows.append((i, j))

    print(f"Matching rows for {mov}:")
    print(matching_rows)
    
    # Horizontal-stack both embeddings to create joint model
    X_joint = np.hstack([embeddings, nuis_regress])
    print(f"Joint predictor matrix shape: {X_joint.shape}")

    width_m1 = embeddings.shape[1]
    width_m2 = nuis_regress.shape[1]

    slice_m1 = slice(0, width_m1)
    slice_m2 = slice(width_m1, width_m1 + width_m2)
    print(f"model 1 slice: {slice_m1}")
    print(f"model 2 slice: {slice_m2}")
    
    # Make pipeline with kernelizer for each feature space
    column_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        Delayer(delays=[2, 3, 4, 5]),
        Kernelizer(kernel="linear"),
    )

    # Compile joint column kernelizer
    column_kernelizer = ColumnKernelizer(
        [('model1', column_pipeline, slice_m1),
         ('model2', column_pipeline, slice_m2)])
    
    
    # Chain transfroms and estimator into pipeline
    pipeline = make_pipeline(column_kernelizer, banded_ridge)
    
    for run in runs:
        for data_type in data_types:
            file_name = os.path.join(data_dir, f"{data_type}_run_{run}_{mov}_data.npy")
            try:
                data = np.load(file_name, allow_pickle=True).item()
            except FileNotFoundError:
                print(f"File {file_name} not found. Skipping...")
                continue
          
            all_subjects = []
            
            for train_index, test_index in loo.split(data):
                print(f'Training on subjects {train_index}, testing on subject {test_index}')

                train_Y = np.vstack([data[idx] for idx in train_index])
                train_X = np.vstack([embeddings for _ in train_index])  # embeddings because we don't need to deal with lags

                test_Y = data[test_index[0]]
                test_X = embeddings # embeddings because we don't need to deal with lags
                
                # Fit the model and predict
                pipeline.fit(train_X, train_Y)
                predicted = pipeline.predict(test_X)
                
                # Compute correlation
                r_parcels = correlation_score(test_Y, predicted)
                all_subjects.append(r_parcels)

            matrix_data = np.array(all_subjects)
            average_across_subjects = np.mean(matrix_data, axis=0)

            r_img = np.zeros(atlas_img.shape)
            for i, parcel in enumerate(np.unique(atlas_img)[1:]):
                r_img[atlas_img == parcel] = average_across_subjects[i]

            r_nii = nib.Nifti1Image(r_img, atlas_nii.affine, atlas_nii.header)
            
            result_file_name = os.path.join(results_dir, f"{data_type}_run_{run}_{mov}_results.npy")
            np.save(result_file_name, average_across_subjects)
            print(f"Saved results to {result_file_name}")

            
## SAVE PDF             

vmax = .6
threshold = .1

# Create a PdfPages object to write multiple plots to a single PDF
for mov in movies:
    # Specify the path for the PDF
    pdf_path = os.path.join(results_dir, f"model1_{mov}_results_banded_ridge_stacked.pdf")  
    with PdfPages(pdf_path) as pdf:
        for data_type in data_types:
            for run in runs:
                file_name = os.path.join(results_dir, f"{data_type}_run_{run}_{mov}_results.npy")
                
                # Check if the file exists and load it
                if os.path.exists(file_name):
                    data = np.load(file_name)

                    # Reshape the data back to the 3D brain shape or 2D cortical surface
                    r_img = np.zeros(atlas_img.shape)  # Initialize an array the same shape as the atlas
                
                    # Map the 1D data back onto the r_img (this step is highly dependent on your specific atlas and data)
                    for i, parcel in enumerate(np.unique(atlas_img)[1:]):
                        r_img[atlas_img == parcel] = data[i]

                    # Convert the 3D array back to a NIfTI image
                    r_nii = nib.Nifti1Image(r_img, atlas_nii.affine, atlas_nii.header)

                    # Create a figure with 2 subplots (side by side)
                    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Plot superior temporal cortex on the first subplot
                    plot_stat_map(r_nii, axes=axes[0], cmap='RdYlBu_r', vmax=vmax, threshold=threshold,
                                  title='Superior Temporal Cortex',
                                  cut_coords=(-55, -24, 9), display_mode='ortho')
                    
                    # Plot posterior medial cortex on the second subplot
                    plot_stat_map(r_nii, axes=axes[1], cmap='RdYlBu_r', vmax=vmax, threshold=threshold,
                                  title='Posterior Medial Cortex',
                                  cut_coords=(-5, -60, 30), display_mode='ortho')

                    # Add a main title to the figure
                    plt.suptitle(f'{mov.capitalize()} ({data_type.upper()}, Run {run})')

                    # Save the current figure into the PDF
                    pdf.savefig(fig)
                    
                    # Close the figure to free memory
                    plt.close(fig)
                    
                    print(f"All plots have been saved into {pdf_path}")

                else:
                    print(f"File {file_name} not found.")

            
# #### References
# 
# * Huth, A. G., De Heer, W. A., Griffiths, T. L., Theunissen, F. E., & Gallant, J. L. (2016). Natural speech reveals the semantic maps that tile human cerebral cortex. *Nature*, *532*(7600), 453-458. https://doi.org/10.1038/nature17637
# 
# * Nastase, S. A., Liu, Y.-F., Hillman, H., Zadbood, A., Hasenfratz, L., Keshavarzian, N., Chen, J., Honey, C. J., Yeshurun, Y., Regev, M., Nguyen, M., Chang, C. H. C., Baldassano, C., Lositsky, O., Simony, E., Chow, M. A., Leong, Y. C., Brooks, P. P., Micciche, E., Choe, G., Goldstein, A., Vanderwal, T., Halchenko, Y. O., Norman, K. A., & Hasson, U. (2020). Narratives: fMRI data for evaluating models of naturalistic language comprehension. *bioRxiv*. https://doi.org/10.1101/2020.12.23.424091
# 
# * Pennington, J., Socher, R., & Manning, C. D. (2014, October). GloVe: Global Vectors for Word Representation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1532-1543). https://www.aclweb.org/anthology/D14-1162
# 
# * Schaefer, A., Kong, R., Gordon, E. M., Laumann, T. O., Zuo, X. N., Holmes, A. J., ... & Yeo, B. T. (2018). Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. Cerebral cortex, 28(9), 3095-3114. https://doi.org/10.1093/cercor/bhx179
