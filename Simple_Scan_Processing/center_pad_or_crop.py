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

def main(argv):
    parser = argparse.ArgumentParser(description="Pad input scan to specified dimension, entered as a ixjxk string.", add_help = True)

    req_group = parser.add_argument_group("Required arguments")
    req_group.add_argument("-i", "--input_dir", help = "input file directory (including filename)", required = True)
    req_group.add_argument("-dim", "--dimension", help = "desired dimension", required = True)
    req_group.add_argument("-o", "--output_dir", help = "output file directory (including filename)", required = True)

    args = parser.parse_args()

    input_scan_nii = nib.load(args.input_dir)
    input_scan_affine = input_scan_nii.affine
    input_scan_header = input_scan_nii.header

    input_dimension = input_scan_nii.shape
    desired_dimension = np.array([int(item) for item in args.dimension.split('x')])
    assert len(desired_dimension) == 3
    print('Input dimension: ', input_dimension, '\ndesired dimension: ', desired_dimension)

    x_lowerbound_target = int(np.floor((desired_dimension[0] - input_dimension[0]) / 2)) if desired_dimension[0] >= input_dimension[0] else 0
    y_lowerbound_target = int(np.floor((desired_dimension[1] - input_dimension[1]) / 2)) if desired_dimension[1] >= input_dimension[1] else 0
    z_lowerbound_target = int(np.floor((desired_dimension[2] - input_dimension[2]) / 2)) if desired_dimension[2] >= input_dimension[2] else 0
    x_upperbound_target = x_lowerbound_target + input_dimension[0] if desired_dimension[0] >= input_dimension[0] else None
    y_upperbound_target = y_lowerbound_target + input_dimension[1] if desired_dimension[1] >= input_dimension[1] else None
    z_upperbound_target = z_lowerbound_target + input_dimension[2] if desired_dimension[2] >= input_dimension[2] else None

    x_lowerbound_input = 0 if desired_dimension[0] >= input_dimension[0] else int(np.floor((input_dimension[0] - desired_dimension[0]) / 2))
    y_lowerbound_input = 0 if desired_dimension[1] >= input_dimension[1] else int(np.floor((input_dimension[1] - desired_dimension[1]) / 2))
    z_lowerbound_input = 0 if desired_dimension[2] >= input_dimension[2] else int(np.floor((input_dimension[2] - desired_dimension[2]) / 2))
    x_upperbound_input = None if desired_dimension[0] >= input_dimension[0] else x_lowerbound_input + desired_dimension[0]
    y_upperbound_input = None if desired_dimension[1] >= input_dimension[1] else y_lowerbound_input + desired_dimension[1]
    z_upperbound_input = None if desired_dimension[2] >= input_dimension[2] else z_lowerbound_input + desired_dimension[2]

    input_scan = np.float64(input_scan_nii.get_fdata())

    output_scan = np.zeros(desired_dimension)

    output_scan[x_lowerbound_target : x_upperbound_target, \
                y_lowerbound_target : y_upperbound_target, \
                z_lowerbound_target : z_upperbound_target] = \
    input_scan[x_lowerbound_input: x_upperbound_input, \
               y_lowerbound_input: y_upperbound_input, \
               z_lowerbound_input: z_upperbound_input]

    output_scan_nifti = nib.Nifti1Image(output_scan, input_scan_affine, input_scan_header)
    nib.save(output_scan_nifti, args.output_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
