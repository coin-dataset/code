#!/usr/bin/python3

"""
sys.argv[1] - input database file
sys.argv[2] - output mat file

Composed by Danyang Zhang @THU_IVG
Last revision: Danyang Zhang @THU_IVG @Oct 3rd, 2019 CST
"""

import json
import scipy.io as sio

import numpy as np
import itertools

import sys

db_f = sys.argv[1]

with open(db_f) as f:
    database = json.load(f)["database"]

steps = list(sorted(set(itertools.chain.from_iterable(
    int(an["id"]) for an in itertools.chain.from_iterable(
        v["annotation"] for v in database.values())))))

min_id = steps[0]
nb_step = len(steps)

init_dist = np.zeros((nb_step,))
frequency_mat = np.zeros((nb_step, nb_step))

for v in database:
    if database[v]["subset"]!="training":
        continue
    for i, an in enumerate(database[v]["annotation"]):
        if i==0:
            init_dist[int(an["id"])-min_id] += 1
        else:
            frequency_mat[int(pan["id"])-min_id, int(an["id"])-min_id] += 1
        pan = an

normalized_init_dist = init_dist/np.sum(init_dist)

frequency_mat_sum = np.sum(frequency_mat, axis=1)
normalized_frequency_mat = np.copy(frequency_mat)
mask = frequency_mat_sum!=0
normalized_frequency_mat[mask] /= frequency_mat_sum[mask][:, None]
zero_position = np.where(np.logical_not(mask))[0]
normalized_frequency_mat[zero_position, zero_position] = 1.


sio.savemat(sys.argv[2], {
    "init_dist": init_dist,
    "frequency_mat": frequency_mat,

    "normalized_init_dist": normalized_init_dist,
    "normalized_frequency_mat": normalized_frequency_mat,
})
