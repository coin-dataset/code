#!/usr/bin/python3

"""
Generate the constraints matrix of the label lexicon.

Last revision: Danyang Zhang @THU_IVG @Mar 6th, 2019 CST
"""

import numpy as np
import json
import sys

json_file = sys.argv[1]
npy_file = sys.argv[2]

with open(json_file) as f:
	database = json.load(f)["database"]

label_set = list(sorted(set(database[v]["class"] for v in database))) # the set of the task labels
action_set = set() # the set of the action labels
for v in database:
	action_set |= set(int(an["id"]) for an in database[v]["annotation"])
action_set = list(sorted(action_set))
label_count = len(label_set) # the number of the task labels
action_count = action_set[-1] # the number of the action labels
matrix = np.zeros((label_count,action_count))

for v in database:
	for an in database[v]["annotation"]:
		tag_id = int(an["id"])
		matrix[label_set.index(database[v]["class"])][tag_id] = 1

np.save(npy_file,matrix)
