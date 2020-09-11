#!/home/raphael/anaconda3/envs/HumanCBV/bin/python3.7
# -*- coding: utf-8 -*-
"""
This script is to check unique values in Nifti files
Copyright: Nanyan "Rosalie" Zhu

"""
import nibabel as nib 
import numpy as np
import os 
import sys
import argparse

def main(argv):
    parser = argparse.ArgumentParser(description="Get Nifti unique value.", add_help=False)

    req_group = parser.add_argument_group("Required arguments")
    req_group.add_argument("-i", "--input_dir", help = "input file directory (including filename)", required = True)
    args = parser.parse_args()
    input_file = args.input_dir
    input_scan = nib.load(input_file).get_fdata()
    
    print(np.unique(input_scan))

if __name__ == "__main__":
    main(sys.argv[1:])
