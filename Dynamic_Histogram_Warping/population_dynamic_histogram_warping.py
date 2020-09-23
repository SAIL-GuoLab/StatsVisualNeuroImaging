#!/home/raphael/anaconda3/envs/HumanCBV/bin/python3.7
# -*- coding: utf-8 -*-

"""
This script implements (to some extent, hopefully) the idea of dynamic histogram warping, introducted in Roy, Cox and Hingorani in their 1995 paper, Dynamic Histogram Warping of Image Pairs for Constant Image Brightness.

For this particular implementation, we aim to reduce the sequence-specific appearance variance across two different cohorts while retaining the between-subject appearance variance.
Specifically, we approach this in the following manner.
1. Find the mean population histogram for cohort A. (mean of WB-top10%mean-normalized brain-region histograms for all subjects in cohort A).
2. Find the mean population histogram for cohort B.
3. Use dynamic histogram warping (implemented with the dynamic time warping algorithm) to warp the first histogram to the second histogram.
4. Apply the same histogram transform to all subjects in cohort B.

Note: since there is a normalization step, the relative scaling among subjects will be affected.

Chen "Raphael" Liu and Nanyan "Rosalie" Zhu
"""

import sys
import argparse
import nibabel as nib
import os
from matplotlib import pyplot as plt
from glob import glob
import numpy as np
from fastdtw import fastdtw
from tqdm import tqdm

def main(argv):
    parser = argparse.ArgumentParser(description="Dynamic histogram warping for nifti scans.", add_help = True)

    req_group = parser.add_argument_group("Required arguments")
    req_group.add_argument("-i", "--input_dir", help = "input population folder (please end with '/')", required = True)
    req_group.add_argument("-mi", "--input_mask_dir", help = "mask file folder corresponding to the input (please end with '/')", required = True)
    req_group.add_argument("-ih", "--input_histogram_dir", help = "input population histogram directory with file name end with .npy", required = True)
    req_group.add_argument("-rh", "--reference_histogram_dir", help = "reference histogram folder (please end with '/')", required = True)
    req_group.add_argument("-o", "--output_dir", help = "output file folder (please end with '/')", required = True)
 
    opt_group = parser.add_argument_group("Optional arguments")
    #opt_group.add_argument("-dec", "--num_decimals", help = "the number of decimal places to round the input and reference scans to", default = 2)
    opt_group.add_argument("-bins", "--bins", help = "the number of bins for the histogram of the input and reference scans", default = 1024)
    opt_group.add_argument("-debug", "--debug", help = "whether or not to print the warping intensity pairs", default = 0)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok = True)
    # load input and target population histogram
    input_mean_histogram = np.load(args.input_histogram_dir + 'mean_histogram.npy')
    bin_edges = np.load(args.input_histogram_dir + 'bin_edges.npy')
    print('Done loading the population mean histogram for the input cohort.')

    reference_mean_histogram = np.load(args.reference_histogram_dir + 'mean_histogram.npy')
    print('Done loading the population mean histogram for the reference cohort.')

    # glob input scans
    input_paths = sorted(glob(args.input_dir + '*.nii.gz'))
    input_mask_paths = list(np.sort(glob(args.input_mask_dir + '*.nii.gz')))

    print(f'{str(len(input_paths))} input scans loaded from {args.input_dir}')

    # Use fastdtw to calculate the mappings between the two histograms.
    distance, path = fastdtw(input_mean_histogram, reference_mean_histogram)

    print('DHW warping method calculated. Start matching individual input scans.')

    # Iterate over all input scans perform dynamic histogram warping.
    for scan_index in tqdm(range(len(input_paths))):
        input_scan = np.float64(nib.load(input_paths[scan_index]).get_fdata())
        input_scan_name = input_paths[scan_index].split('/')[-1]
        matched_scan_name = input_scan_name[:-7] + '_DHW.nii.gz'
        if os.path.exists(args.output_dir + matched_scan_name):
            pass
        else:
            input_mask = np.int16(nib.load(input_mask_paths[scan_index]).get_fdata())
            normalization_factor = np.mean(input_scan[np.logical_and(input_scan >= np.percentile(input_scan[input_mask == 1], 90, interpolation = 'nearest'), input_mask == 1)])
            input_scan_normalized = input_scan / normalization_factor * 256.0

            # Round the scan, in order to reduce the amount of computation
            #input_scan_normalized = np.round(input_scan_normalized, args.num_decimals)

            # Use DHW to generate a matched scan.
            matched_scan = np.zeros(input_scan.shape)
            for current_bin in range(1, len(bin_edges) - 1):
                input_intensity_lower_bound = bin_edges[current_bin - 1]
                input_intensity_upper_bound = bin_edges[current_bin]    
                mapping = [pair for pair in path if pair[0] == current_bin][0]
                matched_voxel_intensity = bin_edges[mapping[1]]
                if str(args.debug) == '1':
                    print(input_intensity_lower_bound, input_intensity_upper_bound, matched_voxel_intensity)
                matched_scan[np.logical_and(input_scan_normalized >= input_intensity_lower_bound, input_mask == 1)] = matched_voxel_intensity
            #matched_scan = np.round(matched_scan, args.num_decimals)

            # Extract the header and affine from one file.
            input_scan_nii = nib.load(input_paths[scan_index])
            input_scan_affine = input_scan_nii.affine
            input_scan_header = input_scan_nii.header
            output_scan_nifti = nib.Nifti1Image(matched_scan, input_scan_affine, input_scan_header)
            nib.save(output_scan_nifti, args.output_dir + matched_scan_name)

if __name__ == "__main__":
    main(sys.argv[1:])
