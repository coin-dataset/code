#!/usr/bin/python3

import numpy as np
import json
import sys

json_file = sys.argv[1]
npy_file = sys.argv[2]

with open(json_file) as f:
	database = json.load(f)["database"]

label_set = list(sorted(set(database[v]["class"] for v in database)))
action_set = set()
for v in database:
	action_set |= set(int(an["id"]) for an in database[v]["annotation"])
action_set = list(sorted(action_set))
label_count = len(label_set)
#action_count = len(action_set)
action_count = action_set[-1]
#min_action_id = action_set[0]
matrix = np.zeros((label_count,action_count))

for v in database:
	for an in database[v]["annotation"]:
		tag_id = int(an["id"])
		matrix[label_set.index(database[v]["class"])][tag_id] = 1

np.save(npy_file,matrix)
