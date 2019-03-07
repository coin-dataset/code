#!/usr/bin/python3

"""
Apply Task-Consistency method to R-C3D results.

Contributed by Danyang Zhang @THU_IVG
Last revision: Danyang Zhang @THU_IVG @Mar 6th, 2019 CST
"""

import json
import numpy as np
import math
import sys

# R-C3D score json
with open(sys.argv[1]) as score_file:
	whole_file = json.load(score_file)
	results = whole_file["results"]

# our info json
with open(sys.argv[2]) as info_file:
	annotations = json.load(info_file)
	database = annotations["database"]

# R-C3D info json
with open(sys.argv[3]) as info_file:
	taxonomy = json.load(info_file)["taxonomy"]

targets = list(set(database[v]["class"] for v in database))
targets.sort()
target_count = len(targets)
# construct the task label set

labels_in_int = [k["nodeID"] for k in taxonomy if k["parentName"]!="Root"]
min_label = min(labels_in_int)
max_label = max(labels_in_int)
label_count = max_label-min_label+1
# construct the action label set

target_label_constraints = [set() for i in range(target_count)]
min_id = 0x7fffffff
for v in database:
	min_id = min(min_id,min(int(an["id"]) for an in database[v]["annotation"]))
for v in database:
	target_num = targets.index(database[v]["class"])
	for an in database[v]["annotation"]:
		target_label_constraints[target_num].add(int(an["id"])-min_id)
# obtain the constraints between the task labels and the action labels

for v in results:
	score_sum = np.zeros((label_count,))
	for prediction in results[v]:
		score_sum[int(prediction["label"])-min_label] += prediction["score"]
	target_score = np.zeros((target_count,))
	for tgt_num,tgt in enumerate(targets):
		for lbl in target_label_constraints[tgt_num]:
			target_score[tgt_num] += score_sum[lbl]
        # aggregate the scores of the action classes under the identical task/target class
	
        probable_target = np.argmax(target_score) # infer the probable task label
	for prediction in results[v]:
		if int(prediction["label"])-min_label not in target_label_constraints[probable_target]:
			prediction["score"] *= math.exp(-2)
        # refine the prediction label

with open(sys.argv[1] + ".new","w") as new_score_file:
	json.dump(whole_file,new_score_file,indent="\t",ensure_ascii=False)
