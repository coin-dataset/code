#!/usr/bin/python3

import json
import evaluate
import sys
import numpy as np
import terminaltables

groundtruth_file = sys.argv[1]
result_file = sys.argv[2]

# read the groundtruths
with open(groundtruth_file) as f:
	groundtruths = json.load(f)
	taxonomy = groundtruths["taxonomy"]
	database = groundtruths["database"]

labels_in_int = [k["nodeID"] for k in taxonomy if k["parentName"]!="Root"]
min_label = min(labels_in_int)
max_label = max(labels_in_int)
label_count = max_label-min_label+1
evaluate.number_label = label_count

groundtruth_by_cls = [[] for i in range(label_count)]
all_groundtruth = []
for v in database:
	if database[v]["subset"]=="training":
		continue
	for an in database[v]["annotations"]:
		cls = int(an["label"])-min_label
		groundtruth_by_cls[cls].append([cls,an["segment"][0],an["segment"][1],1,v])
for cls in groundtruth_by_cls:
	all_groundtruth += cls

print("Groundtruths read in.")

# read the results
with open(result_file) as f:
	results = json.load(f)["results"]

top_k = 60

prediction_by_cls = [[] for i in range(label_count)]
all_prediction = []
for v in results:
	results[v].sort(key=(lambda k:k["score"]),reverse=True)
	for i,prediction in enumerate(results[v]):
		if i>=top_k:
			break
		cls = int(prediction["label"])-min_label
		prediction_by_cls[cls].append([cls,prediction["segment"][0],prediction["segment"][1],prediction["score"],v])

print("Results read in.")

# perform nms
nms_threshold = 0.6
nmsed_prediction_by_cls = [[] for i in range(label_count)]
for cls in prediction_by_cls:
	cls.sort(key=(lambda v: v[3]),reverse=True)
for cls,pred_grp in enumerate(prediction_by_cls):
	for pred in pred_grp:
		remained_or_not = True
		for r in nmsed_prediction_by_cls[cls]:
			if r[4]==pred[4]:
				intersection = max(0,min(r[2],pred[2])-max(r[1],pred[1]))
				union = max(r[2],pred[2])-min(r[1],pred[1])
				remained_or_not = intersection/union<nms_threshold
		if remained_or_not:
			nmsed_prediction_by_cls[cls].append(pred)
prediction_by_cls = nmsed_prediction_by_cls
for cls in prediction_by_cls:
	all_prediction += cls

print("NMS performed.")

# calculate
ious = np.arange(0.1,1.0,0.1)
aps = np.zeros((len(ious),label_count))
ars = np.zeros((len(ious),label_count))
f1s = np.zeros((len(ious),))

miou = evaluate.miou_per_v(all_prediction,all_groundtruth)

for i,iou in enumerate(ious):
	for cls in range(label_count):
		ap,ar = evaluate.ap(prediction_by_cls[cls],iou,groundtruth_by_cls[cls])
		aps[i][cls] = ap
		ars[i][cls] = ar
	f1 = evaluate.f1(all_prediction,iou,all_groundtruth)
	f1s[i] = f1

map_ = np.mean(aps,axis=1)
mar = np.mean(ars,axis=1)

print("Criteria solved.")

# print
title = "C3D Detection Performance"
datas = [["IoU threshold"], ["mean AP"], ["mean AR"], ["F1 criterion"]]
for i,iou in enumerate(ious):
	datas[0].append("{:.2f}".format(iou))
	datas[1].append("{:.4f}".format(map_[i]))
	datas[2].append("{:.4f}".format(mar[i]))
	datas[3].append("{:.4f}".format(f1s[i]))

datas[0].append("Average")
datas[1].append("{:.4f}".format(np.mean(map_)))
datas[2].append("{:.4f}".format(np.mean(mar)))
datas[3].append("{:.4f}".format(np.mean(f1s)))
table = terminaltables.AsciiTable(datas,title)
table.justify_columns[-1] = "right"
table.inner_row_border = True
print(table.table)
print("mIou: {:.4f}".format(miou))
