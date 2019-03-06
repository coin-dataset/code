#!/usr/bin/python3

"""
Transfer the pkl scores to npy.

Contributed by Danyang Zhang @THU_IVG
Last revision: Danyang Zhang @THU_IVG @Mar 6th, 2019 CST
"""

import numpy as np
import json
import os
import os.path
import pickle
import sys
import collections
import math

with open(sys.argv[1],"rb") as score_file:
	scores = pickle.load(score_file)

output_prefix = sys.argv[1][0:sys.argv[1].rfind(".")]
try:
	os.makedirs(output_prefix)
except OSError:
	pass

with open(sys.argv[2]) as info_file:
	annotations = json.load(info_file)["database"]

for v in scores:
	vid = v.split("/")[-1]
	video_duration = annotations[vid]["duration"]

	proposals = scores[v][0]
	actionness = scores[v][1]
	completeness = scores[v][2]
	regression = scores[v][3]

	score_max = np.max(actionness,axis=-1)
	exp_score = np.exp(actionness-score_max[...,None])
	exp_com = np.exp(completeness)
	combined_scores = (exp_score/np.sum(exp_score,axis=-1)[...,None])[:,1:]*exp_com
        # combined scores are calculated as softmax(actionness)*exp(completeness) according to the code offered by SSN

	proposal_count = len(proposals)
	class_count = completeness.shape[1]
	proposal_npy = np.zeros((proposal_count,class_count,7))
        # the columns in proposal_npy: 
        # start of the proposal range, end of the proposal range, exp(actionness), exp(completeness), combined score, actionness, completeness

	for i in range(proposal_count):
		start = proposals[i][0]*video_duration
		end = proposals[i][1]*video_duration

		for c in range(class_count):
			proposal_npy[i][c][0] = proposals[i][0]
			proposal_npy[i][c][1] = proposals[i][1]
			proposal_npy[i][c][2] = exp_score[i][c+1]
			proposal_npy[i][c][3] = exp_com[i][c]
			proposal_npy[i][c][4] = combined_scores[i][c]
			proposal_npy[i][c][5] = actionness[i][c+1]
			proposal_npy[i][c][6] = completeness[i][c]
	npy_name = os.path.join(output_prefix,"proposal_" + vid)
	np.save(npy_name,proposal_npy)
