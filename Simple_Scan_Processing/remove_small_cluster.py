#!/home/raphael/anaconda3/envs/HumanCBV/bin/python3.7
# -*- coding: utf-8 -*-
"""
This is a function to remove small clusters or scattered points. 
Copyright: Nanyan "Rosalie" Zhu & Chen "Raphael" Liu
"""
#%%import packages
import numpy as np
import nibabel as nib
from glob import glob
from skimage.measure import label, regionprops
import sys
import argparse
import os

def main(argv):
    parser = argparse.ArgumentParser(description = "Remove small clusters.", add_help = False)
    req_group = parser.add_argument_group("Required arguments")
    req_group.add_argument("-i", "--input_dir", help = "input directory",required = True)
    req_group.add_argument("-c", "--cluster_size", help = "cluster size below which will be cleaned", required = True)
    req_group.add_argument("-o", "--output_dir", help = "output directory of the generated t-map", required = True)
    
    args = parser.parse_args()
    
    input_scan_nii = nib.load(args.input_dir)
    affine = input_scan_nii.affine
    header = input_scan_nii.header
    input_scan = np.float32(input_scan_nii.get_fdata())

    # Find the biggest 3D connected component.
    label_map = label(input_scan != 0)
    region_area = []; region_label = []
    for region in regionprops(label_map):
        region_area.append(region.area)
        region_label.append(region.label)
    assert len(region_area) == len(region_label)
    assert len(region_area) > 0

    print('cluster sizes: ', region_area)
    region_labels_large_enough = np.array(region_label)[np.array(region_area) >= np.int16(args.cluster_size)]
    
    # Only keep the biggest 3D connected component.
    output_scan = np.float64(np.zeros(input_scan.shape))
    for current_cluster_label in np.unique(region_labels_large_enough):
        output_scan[label_map == current_cluster_label] = input_scan[label_map == current_cluster_label]

    output_scan_nii = nib.Nifti1Image(output_scan, affine, header)
    if len(args.output_dir.split('/')) > 1:
        os.makedirs('/'.join(args.output_dir.split('/')[:-1]), exist_ok = True)
    nib.save(output_scan_nii, args.output_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
