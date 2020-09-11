#!/home/raphael/anaconda3/envs/HumanCBV/bin/python3.7
# -*- coding: utf-8 -*-
"""
This function is to convert brain extracted scans to brain mask scans.
Now it only works for '.nii.gz' scans.
Authors: Nanyan "Rosalie" Zhu and Chen "Raphael" Liu

"""
import nibabel as nib 
import numpy as np
import os 
import sys
import argparse
from skimage.measure import label, regionprops
from scipy.ndimage.morphology import binary_fill_holes

def main(argv):
    parser = argparse.ArgumentParser(description = "Convert brain extracted scans to brain mask scans.", add_help = False)

    req_group = parser.add_argument_group("Required arguments")
    req_group.add_argument("-i", "--input_file", help = "Input file name (required)", required = True)
    req_group.add_argument("-o", "--output_file", help = "Output file name (required)", required = True)

    opt_group = parser.add_argument_group("Optional arguments")
    opt_group.add_argument("-t", "--threshold", type = float, default = 0, help = "The threshold level for the binarization of the WB scan. Values less than or equal to threshold are replaced by zeros.", required = False)

    args = parser.parse_args()

    input_file = args.input_file
    if input_file.split('.')[-1] == 'gz' and input_file.split('.')[-2] == 'nii':
    	print('This is the filetype that we approve: ".nii.gz".')
    else:
    	print('This function currently only works for ".nii.gz" files.')

    # Load the nifti scan.
    input_scan_nifti = nib.load(input_file)
    input_scan = np.float32(input_scan_nifti.get_fdata())
    affine = input_scan_nifti.affine
    header = input_scan_nifti.header

    # Binarization at the threshold given. If not given, binarize at 0.
    binary_scan = input_scan.copy()
    binary_scan[input_scan <= np.float32(args.threshold)] = 0
    binary_scan[input_scan > np.float32(args.threshold)] = 1

    # Binary hole filling.
    filled_scan = np.float32(binary_fill_holes(binary_scan))

    # Discard everything except for the biggest 3D connected component.
    label_map = label(filled_scan)
    region_area = []; region_label = []
    for region in regionprops(label_map):
        region_area.append(region.area)
        region_label.append(region.label)
    assert len(region_area) == len(region_label)
    assert len(region_area) > 0
   
    region_label_biggest_component = region_label[np.argmax(region_area)]
    
    new_scan = np.float32(np.zeros(filled_scan.shape))
    new_scan[label_map == region_label_biggest_component] = 1
    new_scan_nifti = nib.Nifti1Image(new_scan, affine, header)
    
    nib.save(new_scan_nifti, args.output_file)

    print("If your brain mask turns out to be empty, please try a different threshold.")

if __name__ == "__main__":
    main(sys.argv[1:])
