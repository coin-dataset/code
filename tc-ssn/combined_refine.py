#!/usr/bin/python3

"""
Refine the scores combined from actionness and completeness scores outputed by SSN.

Last revision: Danyang Zhang @THU_IVG @Mar 6th, 2019 CST
"""

import numpy as np
import os
import os.path
import math
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--constraints","-c",action="store",type=str,required=True)
parser.add_argument("--src-score","-i",action="store",type=str,required=True)
parser.add_argument("--target","-o",action="store",type=str,default="test_gt_score_combined_refined_fusion")
args = parser.parse_args()

constraints = np.load(args.constraints) # constraints matrix
target_class_count,action_class_count = constraints.shape

numpy_dir = args.src_score
target_dir = args.target

try:
	os.makedirs(target_dir)
except OSError:
	pass

numpys = os.listdir(numpy_dir)
for np_file in numpys:
	if np_file.endswith("_groundtruth.npy"):
		continue
	vid = np_file[np_file.find("_")+1:np_file.rfind(".")]
	premat = np.load(os.path.join(numpy_dir,np_file))
	combined = premat[:,:,4]
	video_combined = np.sum(combined,axis=0)
	target_class_combined = np.zeros((target_class_count,))
	for target_cls in range(target_class_count):
		for act_cls in range(action_class_count):
			if constraints[target_cls][act_cls]==1:
				target_class_combined[target_cls] += video_combined[act_cls]
        # aggregate the scores of the action classes under the identical task/target class
	probable_target_class = np.argmax(target_class_combined) # infer the probable task class
	mask = np.full(combined.shape,math.exp(-2))
	mask[:,0] = 1
	mask[:,np.where(constraints[probable_target_class])[0]] = 1
	combined *= mask
        # refine the combined scores
	premat[:,:,4] = combined
	np.save(os.path.join(target_dir,np_file),premat)
