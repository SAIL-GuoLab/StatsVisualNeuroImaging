#!/home/raphael/anaconda3/envs/HumanCBV/bin/python3.7
# -*- coding: utf-8 -*-
"""
This function plots three slices each along the axial, saggital and coronal directions and saves the result as a png image. Won't guarantee the order.

Copyright: Chen "Raphael" Liu & Nanyan "Rosalie" Zhu

"""
import nibabel as nib 
from matplotlib import pyplot as plt
import numpy as np
import os 
import sys
import argparse
from scipy.ndimage import rotate

def main(argv):
    parser = argparse.ArgumentParser(description = "Plot slices from scan.", add_help = False)
    req_group = parser.add_argument_group("Required arguments")
    req_group.add_argument("-i", "--input_file", help = "Input file name (required)", required = True)

    opt_group = parser.add_argument_group("Optional arguments")
    opt_group.add_argument("-o", "--output_file", help = "Output file name (if not specified, will use the input file name)")
    opt_group.add_argument("-oo", "--output_folder")
    opt_group.add_argument("-c", "--colormap")

    args = parser.parse_args()

    input_file = args.input_file
    if input_file.split('.')[-1] == 'gz' and input_file.split('.')[-2] == 'nii':
    	print('This is the filetype that we approve: ".nii.gz".')
    else:
    	error('This function currently only works for ".nii.gz" files.')

    # Load the nifti scan.
    input_scan = np.float32(nib.load(input_file).get_fdata())

    colormap = 'gray' if args.colormap == None else str(args.colormap)

    # Plot the slices.
    # The order and orientation are optimized for the MNI template.
    plt.rcParams['figure.figsize'] = [45, 8]
    plt.subplot(1, 9, 1)
    plt.imshow(rotate(input_scan[:, :, input_scan.shape[2]*2//3], 90, order = 0), cmap = colormap)
    plt.axis('off')
    plt.subplot(1, 9, 2)
    plt.imshow(rotate(input_scan[:, :, input_scan.shape[2]//2], 90, order = 0), cmap = colormap)
    plt.axis('off')
    plt.subplot(1, 9, 3)
    plt.imshow(rotate(input_scan[:, :, input_scan.shape[2]//3], 90, order = 0), cmap = colormap)
    plt.axis('off')
    plt.subplot(1, 9, 4)
    plt.imshow(rotate(input_scan[input_scan.shape[0]//3, :, :], 90, order = 0), cmap = colormap)
    plt.axis('off')
    plt.subplot(1, 9, 5)
    plt.imshow(rotate(input_scan[input_scan.shape[0]//2, :, :], 90, order = 0), cmap = colormap)
    plt.axis('off')
    plt.subplot(1, 9, 6)
    plt.imshow(rotate(input_scan[input_scan.shape[0]*2//3, :, :], 90, order = 0), cmap = colormap)
    plt.axis('off')
    plt.subplot(1, 9, 7)
    plt.imshow(rotate(input_scan[:, input_scan.shape[1]*2//3, :], 90, order = 0), cmap = colormap)
    plt.axis('off')
    plt.subplot(1, 9, 8)
    plt.imshow(rotate(input_scan[:, input_scan.shape[1]//2, :], 90, order = 0), cmap = colormap)
    plt.axis('off')
    plt.subplot(1, 9, 9)
    plt.imshow(rotate(input_scan[:, input_scan.shape[1]//3, :], 90, order = 0), cmap = colormap)
    plt.axis('off')
    
    # Save the figure.
    if args.output_file is None:
        if args.output_folder is None:
            args.output_file = input_file[:-7] + '.png'
        else:
            args.output_file = args.output_folder + input_file.split('/')[-1][:-7] + '.png'
            os.makedirs(args.output_folder, exist_ok = True)
    plt.tight_layout()

    plt.savefig(args.output_file, facecolor = 'black')

if __name__ == "__main__":
    main(sys.argv[1:])
