#!/home/sail/.conda/envs/MouseBrain/bin/python3.7
# -*- coding: utf-8 -*-

"""
This script pads or crops a MRI scan to a given dimension. The original matrix will be centered in the specified new dimension. New voxels will be zero padded.

Nanyan "Rosalie" Zhu
"""
import sys
import argparse
import nibabel as nib
import os
from glob import glob
import numpy as np
from tqdm import tqdm

def main(argv):
    parser = argparse.ArgumentParser(description="Compute mean scan. Each voxl represents the mean of corresponding voxels from all input scans", add_help = True)

    req_group = parser.add_argument_group("Required arguments")
    req_group.add_argument("-i", "--input_dir", help = "input file directory (folder name)", required = True)
    req_group.add_argument("-o", "--output_dir", help = "output file directory (path and filename)", required = True)

    args = parser.parse_args()

    input_scan_paths = list(np.sort(glob(args.input_dir + '/*.nii.gz')))
    print('We are going to take the mean over %s scans.' % len(input_scan_paths))
    for scan_index in tqdm(range(len(input_scan_paths))):
        if scan_index == 0:
            input_scan_nii = nib.load(input_scan_paths[scan_index])
            input_scan_affine = input_scan_nii.affine
            input_scan_header = input_scan_nii.header
            population_mean_scan = np.zeros(input_scan_nii.shape)
        input_scan = np.float64(nib.load(input_scan_paths[scan_index]).get_fdata())
        population_mean_scan += input_scan

    population_mean_scan /= len(input_scan_paths)
    population_mean_scan_nifti = nib.Nifti1Image(population_mean_scan, input_scan_affine, input_scan_header)
    nib.save(population_mean_scan_nifti, args.output_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
