#!/home/raphael/anaconda3/envs/HumanCBV/bin/python3.7
# -*- coding: utf-8 -*-
"""
This is a function to get two cohort ttest tmap (cohort A > cohort B). 
Specifically, this is written for special cases where the input cohorts are very large (e.g., > 1000 scans in total).
We use special means to reduce memory consumption.
Copyright: Nanyan "Rosalie" Zhu & Chen "Raphael" Liu
"""
#%%import packages
import numpy as np
import nibabel as nib
from glob import glob
from scipy.stats import ttest_ind
import sys
import argparse
import os
from tqdm import tqdm
import h5py

def main(argv):
    #%% Define the input argument parser to handle input arguments.
    parser = argparse.ArgumentParser(description = "Get two cohort t-test t-score map for cohort A > cohort B.", add_help = False)
    req_group = parser.add_argument_group("Required arguments")
    req_group.add_argument("-A", "--cohort_A_folder", \
        help = "Input cohort A folder (required). Make sure all files (.nii.gz) are kept in this folder, not in subfolders.",required = True)
    req_group.add_argument("-B", "--cohort_B_folder", \
        help = "Input cohort B folder (required). Make sure all files (.nii.gz) are kept in this folder, not in subfolders.",required = True)
    req_group.add_argument("-o", "--output_dir", \
        help = "output directory of the generated t-map (including filename).", required = True)

    opt_group = parser.add_argument_group("Optional arguments")
    opt_group.add_argument("-p", "--p_thr", \
        help = "p value threshold. Default (0.5) will not throw away any valid t-score point.", default = 0.5)
    opt_group.add_argument("-m", "--FOV_dir", \
        help = "mask file directory (including filename)." + \
        " Mask is used to define the field of view only inside which we calculate the t-score.")
    opt_group.add_argument("-pmap", "--generate_p_map", help = "whether or not to generate a p-value map.")
    opt_group.add_argument("-pmap_dir", "--pmap_dir")
    opt_group.add_argument("-pmap_mask_dir", "--pmap_mask_dir")

    #%% Parse the input arguments.
    args = parser.parse_args()

    #%% If we define the field-of-view mask file, we load it into a numpy matrix.
    if not args.FOV_dir is None:
        FOV_mask = nib.load(args.FOV_dir).get_fdata().astype(np.float32)
    
    #%% Grab all files from the two cohorts and store them in a list. The sorting is not necessary in this case but it's a good practice.
    cohort_A_files = list(np.sort(glob(args.cohort_A_folder + '*.nii.gz')))
    cohort_B_files = list(np.sort(glob(args.cohort_B_folder + '*.nii.gz')))
    print('cohort A: %d \ncohort B: %d' % (len(cohort_A_files), len(cohort_B_files)))

    #%% Try to load one scan from each cohort. Assure that their dimensions match.
    print('Trying to load one scan from each cohort >>>')
    cohort_A_first_scan = nib.load(cohort_A_files[0])
    cohort_B_first_scan = nib.load(cohort_B_files[0])
    assert cohort_A_first_scan.shape == cohort_B_first_scan.shape 
    #%% Also make sure the field-of-view scan can align with the scans.
    if not args.FOV_dir is None:
        assert cohort_B_first_scan.shape == FOV_mask.shape
    print('>>> Success!')

    #%% Load all scans in both cohorts.
    print('Pre-allocating two big matrices, one for each cohort >>>')
    # Here we make sure we pre-allocate a big matrix with the correct final dimensions for each cohort.
    # For doing so, we need to first create a h5py file such that we can store matrices too large for RAMs to handle.
    h5py_file_folder = '/'.join(os.path.abspath(args.output_dir).split('/')[:-1]) + '/h5py_matrices/'
    h5py_file_directory = h5py_file_folder + '%s.hdf5' % args.output_dir.split('/')[-1].split('.nii.gz')[0]
    os.makedirs(h5py_file_folder, exist_ok = True)
    # If we have already created the h5py file and the fully filled big matrices are correctly stored, read them.
    if os.path.exists(h5py_file_directory):
        print('\n\nWaringing: We found an existing h5py file in your output directory with the same naming you defined!' + \
            '\nIf you have not run this test before, please double check what happend.' + \
            '\nIf you are re-running this test, please make sure the saved h5py is correct and complete, so that we can directly load it.' + \
            '\nWhat we mean is: if you previously stopped the process AFTER you loaded all scans from both cohorts and WHILE computing the t-score map, you are safe to load the h5py file here.'
            '\nOtherwise, please move the h5py file to somewhere else, re-run this script and we will perform the correct data organization process.')
        hdf5_storage = h5py.File(h5py_file_directory, 'r')
        Big_A_matrix = hdf5_storage['Big_A_matrix']
        Big_B_matrix = hdf5_storage['Big_B_matrix']
        print('\n\nTwo big matrices directly loaded from the existing h5py file.' + \
            '\nPlease Please Please make sure the h5py file is completely filled when created.' + \
            '\nIf you are not sure, please move the h5py file to somewhere else, re-run this script and we will perform the correct data organization process.\n')
    # If we have not or if we are unsure, do the process from the beginning.
    else:
        hdf5_storage = h5py.File(h5py_file_directory, 'a')
        Big_A_matrix = hdf5_storage.create_dataset("Big_A_matrix", \
            (len(cohort_A_files), cohort_A_first_scan.shape[0], cohort_A_first_scan.shape[1], cohort_A_first_scan.shape[2]))
        Big_B_matrix = hdf5_storage.create_dataset("Big_B_matrix", \
            (len(cohort_B_files), cohort_B_first_scan.shape[0], cohort_B_first_scan.shape[1], cohort_B_first_scan.shape[2]))
        print('>>> Success!')

        #%% Iterate over all scans from cohort A and update the values in Big_A_matrix
        print('Trying to load all scans from both cohorts >>>')
        for scan_index in tqdm(range(len(cohort_A_files))):
            cohort_A_current_scan = nib.load(cohort_A_files[scan_index]).get_fdata().astype(np.float32)
            Big_A_matrix[scan_index, :, :, :] = cohort_A_current_scan

        for scan_index in tqdm(range(len(cohort_B_files))):
            cohort_B_current_scan = nib.load(cohort_B_files[scan_index]).get_fdata().astype(np.float32)
            Big_B_matrix[scan_index, :, :, :] = cohort_B_current_scan 
        print('>>> Success!')

    #%% Create empty matrices for the results (t-score map and potentially the p-value map).
    print('Creating empty matrices to store the results >>>')
    tmap_shell = np.zeros(cohort_A_first_scan.shape)
    if not args.generate_p_map is None:
        if int(args.generate_p_map) == 1:
            pmap_shell = np.ones(cohort_A_first_scan.shape).astype(np.float32)
            pmap_mask_shell = np.zeros(cohort_A_first_scan.shape).astype(np.float32)
    result_affine = cohort_A_first_scan.affine
    result_header = cohort_A_first_scan.header
    print('>>> Success!')

    print('Start calculating the t-score map >>>')
    if not args.FOV_dir is None:
        location = np.where(FOV_mask)
        for idx in tqdm(range(len(location[0]))):
            i = location[0][idx]
            j = location[1][idx]
            k = location[2][idx]
            
            cohort_A_array = Big_A_matrix[:, i, j, k]
            cohort_B_array = Big_B_matrix[:, i, j, k]
            t, p = ttest_ind(cohort_A_array, cohort_B_array, equal_var=True)
            if (p / 2) < np.float32(args.p_thr) and t > 0:
                tmap_shell[i, j, k] = t
            if not args.generate_p_map is None:
                if int(args.generate_p_map) == 1 and t > 0:
                    pmap_shell[i, j, k] = p / 2
                    pmap_mask_shell[i, j, k] = 1
    else:
        for i in tqdm(range(tmap_shell.shape[0])):
            for j in range(tmap_shell.shape[1]):
                for k in range(tmap_shell.shape[2]):
                    cohort_A_array = Big_A_matrix[:, i, j, k]
                    cohort_B_array = Big_B_matrix[:, i, j, k]
                    t, p = ttest_ind(cohort_A_array, cohort_B_array, equal_var=True)
                    if (p / 2) < np.float32(args.p_thr) and t > 0:
                        tmap_shell[i, j, k] = t
                    if not args.generate_p_map is None:
                        if int(args.generate_p_map) == 1 and t > 0:
                            pmap_shell[i, j, k] = p / 2
                            pmap_mask_shell[i, j, k] = 1

    print('>>> t-score map calculated. Saving the t-score map >>>')

    tmap_shell_nii = nib.Nifti1Image(tmap_shell, result_affine, result_header)
    if len(args.output_dir.split('/')) > 1:
        os.makedirs('/'.join(args.output_dir.split('/')[:-1]), exist_ok = True)
    nib.save(tmap_shell_nii, args.output_dir)
    if not args.generate_p_map is None:
        if int(args.generate_p_map) == 1:
            if not args.pmap_dir is None and len(args.pmap_dir.split('/')) > 1:
                os.makedirs('/'.join(args.pmap_dir.split('/')[:-1]), exist_ok = True)
            if not args.pmap_mask_dir is None and len(args.pmap_mask_dir.split('/')) > 1:
                os.makedirs('/'.join(args.pmap_mask_dir.split('/')[:-1]), exist_ok = True)
            if not args.pmap_dir is None:
                pmap_shell_nii = nib.Nifti1Image(pmap_shell, result_affine, result_header)
                nib.save(pmap_shell_nii, args.pmap_dir)
            if not args.pmap_mask_dir is None:
                pmap_mask_shell_nii = nib.Nifti1Image(pmap_mask_shell, result_affine, result_header)
                nib.save(pmap_mask_shell_nii, args.pmap_mask_dir)

    print('>>> Job Complete!')


if __name__ == "__main__":
    main(sys.argv[1:])
