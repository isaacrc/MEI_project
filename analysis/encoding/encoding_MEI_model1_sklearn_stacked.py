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

# Define the directory to save results
results_dir = 'model1/model_1_results_sklearn/stacked'

# Check if the directory exists; if not, create it
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Define hyperparameters for Ridge regression
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
inner_cv = KFold(n_splits=2)
ridge = RidgeCV(alphas=alphas,cv = inner_cv, scoring='r2')
loo = LeaveOneOut()    

    
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

    lags = [2, 3, 4]
    embeddings_lag = []
    for lag in lags:
        prepend = np.zeros((lag, glove_dim))
        append = np.zeros((lags[-1] - lag, glove_dim))
        lagged = np.vstack((prepend, embeddings, append))
        embeddings_lag.append(lagged)

    embeddings_lag = np.hstack(embeddings_lag)

    embeddings_lag = zscore(embeddings_lag, axis=0)

    start_trs = 4
    trim_emb = embeddings_lag[start_trs:, :]
    print('trim_emb:', trim_emb.shape)

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

                train_data = np.vstack([data[idx] for idx in train_index])
                train_labels = np.vstack([trim_emb for _ in train_index])

                test_data = data[test_index[0]]
                test_labels = trim_emb

                ridge.fit(train_labels, train_data)
                predicted = ridge.predict(test_labels)

                r_parcels = [pearsonr(test_data[:, i], predicted[:, i])[0] for i in range(test_data.shape[1])]
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
    pdf_path = os.path.join(results_dir, f"model1_{mov}_results_sklearn_stacked.pdf")  
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
