#!/usr/bin/python3

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
	#video_duration = annotations[vid]["end"]-annotations[vid]["start"]
	video_duration = annotations[vid]["duration"]

	proposals = scores[v][0]
	actionness = scores[v][1]
	completeness = scores[v][2]
	regression = scores[v][3]

	score_max = np.max(actionness[:,1:],axis=-1)
	exp_score = np.exp(actionness[:,1:]-score_max[...,None])
	exp_com = np.exp(completeness)
	combined_scores = (exp_score/np.sum(exp_score,axis=-1)[...,None])*exp_com

	proposal_count = len(proposals)
	class_count = completeness.shape[1]
	proposal_npy = np.zeros((proposal_count,class_count,7))
	for i in range(proposal_count):
		start = proposals[i][0]*video_duration
		end = proposals[i][1]*video_duration

		for c in range(class_count):
			center_proportion = (proposals[i][0]+proposals[i][1])/2.
			duration_proportion = proposals[i][1]-proposals[i][0]
			center_proportion += regression[i][c][0]*duration_proportion
			duration_proportion *= math.exp(regression[i][c][1])
			start_proportion = center_proportion-duration_proportion/2.
			end_proportion = center_proportion+duration_proportion/2.
			start_proportion = max(start_proportion,0.)
			start_proportion = min(start_proportion,1.)
			end_proportion = max(end_proportion,0.)
			end_proportion = min(end_proportion,1.)
			#pre_cls["regressed_interval"] = (start_proportion*video_duration,end_proportion*video_duration)

			proposal_npy[i][c][0] = start_proportion*video_duration
			proposal_npy[i][c][1] = end_proportion*video_duration
			proposal_npy[i][c][2] = exp_score[i][c]
			proposal_npy[i][c][3] = exp_com[i][c]
			proposal_npy[i][c][4] = combined_scores[i][c]
			proposal_npy[i][c][5] = actionness[i][c+1]
			proposal_npy[i][c][6] = completeness[i][c]

	npy_name = os.path.join(output_prefix,"proposal_" + vid)
	np.save(npy_name,proposal_npy)
	np.save(npy_name + "_groundtruth",groundtruth_npy)
	#prediction_dict[vid]["prediction_numpy"] = npy_name + ".npy"
	#prediction_dict[vid]["groundtruth_numpy"] = npy_name + "_groundtruth" + ".npy"
