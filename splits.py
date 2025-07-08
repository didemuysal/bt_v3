# splits.py
# This script divides the dataset into 5 folds for cross-validation,
# ensuring that images from the same patient do not appear in different sets.

import os
import h5py
import numpy as np
from typing import List, Tuple

def get_patient_level_splits(data_dir: str, cvind_path: str) -> List[Tuple]:
    """
    Generates 5-fold cross-validation splits.
    
    The dataset uses a cyclic split strategy for each fold:
    - Fold `i` is the test set.
    - Fold `i+1` (or 1 if i=5) is the validation set.
    - The remaining 3 folds are the training set.
    
    Returns:
        A list of tuples, where each tuple contains the train, validation,
        and test files and labels for one fold.
    """
    
    # --- 1. Load the cross-validation indices ---
    # This file tells us which fold (1-5) each image belongs to.
    with h5py.File(cvind_path, "r") as f:
        # .flatten() converts the 2D array to a 1D array
        fold_assignments = f["cvind"][()].flatten()

    # --- 2. Get a sorted list of all image filenames ---
    # It's crucial to sort them numerically (1.mat, 2.mat, ...) so they
    # match the order of the fold_assignments.
    def get_filenumber(filename):
        return int(os.path.splitext(filename)[0])
        
    all_mat_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".mat")],
        key=get_filenumber
    )
    all_mat_files = np.array(all_mat_files)
    
    # --- 3. Get the label for each image ---
    labels = []
    for filename in all_mat_files:
        filepath = os.path.join(data_dir, filename)
        with h5py.File(filepath, "r") as f:
            labels.append(int(f["cjdata"]["label"][0][0]))
    labels = np.array(labels)
    
    # --- 4. Generate the splits for each of the 5 folds ---
    all_splits = []
    for fold_num in range(1, 6): # Loop from 1 to 5
        
        # Determine which folds to use for train, validation, and test
        test_fold = fold_num
        validation_fold = (fold_num % 5) + 1 # Next fold in the cycle
        
        # Training folds are all folds that are not test or validation
        train_folds = [i for i in range(1, 6) if i not in [test_fold, validation_fold]]
        
        # Create boolean masks to select files for each set
        test_mask = (fold_assignments == test_fold)
        validation_mask = (fold_assignments == validation_fold)
        train_mask = np.isin(fold_assignments, train_folds) # isin checks for multiple values
        
        # Create the tuple for this split
        split_data = (
            all_mat_files[train_mask].tolist(), labels[train_mask].tolist(),
            all_mat_files[validation_mask].tolist(), labels[validation_mask].tolist(),
            all_mat_files[test_mask].tolist(), labels[test_mask].tolist()
        )
        all_splits.append(split_data)
        
    return all_splits