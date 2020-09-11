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
from tqdm import tqdm

def main(argv):
    parser = argparse.ArgumentParser(description="Population histogram calculation for nifti scans.", add_help = True)

    req_group = parser.add_argument_group("Required arguments")
    req_group.add_argument("-i", "--input_dir", help = "input population folder (please end with '/')", required = True)
    req_group.add_argument("-o", "--output_dir", help = "output file folder (please end with '/')", required = True)
    req_group.add_argument("-mi", "--input_mask_dir", help = "mask file folder corresponding to the input (please end with '/')", required = True)

    opt_group = parser.add_argument_group("Optional arguments")
    #opt_group.add_argument("-dec", "--num_decimals", help = "the number of decimal places to round the input and reference scans to", default = 2)
    opt_group.add_argument("-bins", "--bins", help = "the number of bins for the histogram of the input and reference scans", default = 512)

    args = parser.parse_args()

    input_paths = list(np.sort(glob(args.input_dir + '*.nii.gz')))
    input_mask_paths = list(np.sort(glob(args.input_mask_dir + '*.nii.gz')))

    assert len(input_paths) == len(input_mask_paths)

    # Iterate over all input scans to find the population mean histogram.
    for scan_index in tqdm(range(len(input_paths))):
        input_scan = np.float64(nib.load(input_paths[scan_index]).get_fdata())
        input_mask = np.int16(nib.load(input_mask_paths[scan_index]).get_fdata())
        normalization_factor = np.mean(input_scan[np.logical_and(input_scan >= np.percentile(input_scan[input_mask == 1], 90, interpolation = 'nearest'), input_mask == 1)])
        input_scan_normalized = input_scan / normalization_factor * 100.0

        ## Round the scan, in order to reduce the amount of computation
        #input_scan_normalized = np.round(input_scan_normalized, args.num_decimals)

        # Compute the histograms.
        histogram_and_bin_edges = plt.hist(input_scan_normalized[input_mask == 1], bins = args.bins, range = [0, 255.0])
        current_histogram = histogram_and_bin_edges[0]
        bin_edges = histogram_and_bin_edges[1]

        if scan_index == 0:
            input_cumulative_histogram = current_histogram
        else:
            input_cumulative_histogram = input_cumulative_histogram + current_histogram

    input_mean_histogram = input_cumulative_histogram * args.bins / input_cumulative_histogram.sum()
    print('Done calculating the population mean histogram for the input cohort.')
    os.makedirs(args.output_dir, exist_ok = True)
    np.save(args.output_dir + 'mean_histogram.npy', input_mean_histogram)
    np.save(args.output_dir + 'bin_edges.npy', bin_edges)


if __name__ == "__main__":
    main(sys.argv[1:])
