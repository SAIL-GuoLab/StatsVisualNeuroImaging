#!/home/raphael/anaconda3/envs/HumanCBV/bin/python3.7
# -*- coding: utf-8 -*-
"""
This is a script to get regreesion t-score maps. 
Copyright: Nanyan "Rosalie" Zhu & Chen "Raphael" Liu
"""
#%%import packages
import numpy as np
import nibabel as nib
from glob import glob
import sys
import argparse
import os
from tqdm import tqdm
import h5py

mask_path = 
p_thr = 
output_dir =
tail = # 'one' or 'two'
info_csv = 
CBV_dir = 

CBV_path = glob(args.CBV_dir + '*')
mask = nib.load(args.mask_path).get_fdata()

# load csv infomation
CU_schiz_info = pd.read_csv(args.info_csv)
subject_ID = CU_schiz_info['subject ID'].to_list()
gender_list = CU_schiz_info['Gender'].to_list()
age_array = CU_schiz_info['Age'].to_numpy()
group_list = CU_schiz_info['study group'].to_list()

# set dummies for regression (if you are using continuous variable, you don't have to do this encoding)
gender_array = np.zeros((len(gender_list)))
for idx in range(len(gender_list)):
    if gender_list[idx] == 'Male':
        gender_array[idx] = 1

group_array = np.zeros((len(group_list)))
for idx in range(len(group_list)):
    if group_list[idx] == 'converter':
        group_array[idx] = 1
    elif group_list[idx] == 'nonconverter':
        group_array[idx] = 1

assert len(CBV_path) == len(subject_ID)

# sort CBV path corresponding to the csv
subject_CBV_list = [CBV_path[idx].split('/')[-1].split('_')[args.name_id] for idx in range(len(CBV_path))]
CBV_sorted_list = [CBV_path[subject_CBV_list.index(subject_ID[idx])] for idx in range(len(subject_ID))]

# create regression matrix
Big_CBV_matrix = np.zeros((len(CBV_sorted_list), *mask.shape))
for i, CBV_file in enumerate(CBV_sorted_list):
    Big_CBV_matrix[i, :, :, :] = nib.load(CBV_file).get_fdata()


location = np.where(mask)
t_map_shell = np.zeros(mask.shape)
for idx in range(len(location[0])):
    i = location[0][idx]
    j = location[1][idx]
    k = location[2][idx]
    
    intensity_array = Big_CBV_matrix[:, i, j, k]
    x_train = np.column_stack((group_array, gender_array, np.float32(age_array / age_array.max())))
    y_train = intensity_array
    
    clf = linear_model.LinearRegression()
    clf.fit(x_train, y_train)

    params = np.append(clf.intercept_, clf.coef_)
    predicted = clf.predict(x_train)

    
    newX = pd.DataFrame({"Constant":np.ones(len(x_train))}).join(pd.DataFrame(x_train))
    MSE = (sum((y_train-predicted)**2))/(len(newX)-len(newX.columns))
    
    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b
    
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]
    
    # threshold the p-value
    if tail == 'one':
        if (p_values[1] / 2) < args.p_thr and ts_b[1] > 0:
            t_map_shell[i, j, k] = ts_b[1]

    elif tail == 'two':
        if (p_values[1]) < args.p_thr and ts_b[1] > 0:
            t_map_shell[i, j, k] = ts_b[1]
    else:
        raise Exception("only one and two are allowed as the options! Please check your tail setting.")
        
tmap_shell_nii = nib.Nifti1Image(t_map_shell, HF_mask_nii.affine, HF_mask_nii.header)
nib.save(tmap_shell_nii, args.output_dir)
