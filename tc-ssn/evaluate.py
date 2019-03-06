"""
Evaluation utilisation function model. Derived from the evaluation code from PKU-MMD (https://github.com/ECHO960/PKU-MMD). Several mistakes and imcompatible with python3 features are corrected.

Last revision: Danyang Zhang @THU_IVG @Mar 6th, 2019 CST
"""

import os
import numpy as np

theta = 0.5 #overlap ratio
number_label = 52

# calc_pr: calculate precision and recall
#	@positive: number of positive proposal
#	@proposal: number of all proposal
#	@ground: number of ground truth
def calc_pr(positive, proposal, ground):
	if (proposal == 0): return 0,0
	if (ground == 0): return 0,0
	return (1.0*positive)/proposal, (1.0*positive)/ground

def overlap(prop, ground):
	l_p, s_p, e_p, c_p, v_p = prop
	l_g, s_g, e_g, c_g, v_g = ground
	if (int(l_p) != int(l_g)): return 0
	if (v_p != v_g): return 0
	return max((min(e_p, e_g)-max(s_p, s_g))/(max(e_p, e_g)-min(s_p, s_g)),0)

# match: match proposal and ground truth
#	@lst: list of proposals(label, start, end, confidence, video_name)
#	@ratio: overlap ratio
#	@ground: list of ground truth(label, start, end, confidence, video_name)
#
#	correspond_map: record matching ground truth for each proposal
#	count_map: record how many proposals is each ground truth matched by 
#	index_map: index_list of each video for ground truth
def match(lst, ratio, ground):
	cos_map = [-1 for x in range(len(lst))]
	count_map = [0 for x in range(len(ground))]
	#generate index_map to speed up
	index_map = [[] for x in range(number_label)]
	for x in range(len(ground)):
		index_map[int(ground[x][0])].append(x)

	for x in range(len(lst)):
		for y in index_map[int(lst[x][0])]:
			if (overlap(lst[x], ground[y]) < ratio): continue
			if cos_map[x]!=-1 and overlap(lst[x], ground[y]) < overlap(lst[x], ground[cos_map[x]]): continue
			cos_map[x] = y
		if (cos_map[x] != -1): count_map[cos_map[x]] += 1
	positive = sum([(x>0) for x in count_map])
	return cos_map, count_map, positive

# f1-score:
#	@lst: list of proposals(label, start, end, confidence, video_name)
#	@ratio: overlap ratio
#	@ground: list of ground truth(label, start, end, confidence, video_name)
def f1(lst, ratio, ground):
	cos_map, count_map, positive = match(lst, ratio, ground)
	precision, recall = calc_pr(positive, len(lst), len(ground))
	print("{:f} {:f}".format(precision,recall))
	try:
		score = 2*precision*recall/(precision+recall)
	except:
		score = 0.
	return score

# Interpolated Average Precision:
#	@lst: list of proposals(label, start, end, confidence, video_name)
#	@ratio: overlap ratio
#	@ground: list of ground truth(label, start, end, confidence, video_name)
#
#	score = sigma(precision(recall) * delta(recall))
#	Note that when overlap ratio < 0.5, 
#		one ground truth will correspond to many proposals
#		In that case, only one positive proposal is counted
def ap(lst, ratio, ground):
	lst.sort(key = lambda x:x[3]) # sorted by confidence
	cos_map, count_map, positive = match(lst, ratio, ground)
	score = 0;
	number_proposal = len(lst)
	number_ground = len(ground)
	old_precision, old_recall = calc_pr(positive, number_proposal, number_ground)
	total_recall = old_recall
 
	for x in range(len(lst)):
		number_proposal -= 1;
		#if (cos_map[x] == -1): continue
		if cos_map[x]!=-1:
			count_map[cos_map[x]] -= 1;
			if (count_map[cos_map[x]] == 0): positive -= 1;

		precision, recall = calc_pr(positive, number_proposal, number_ground)   
		score += old_precision*(old_recall-recall)
		if precision>old_precision: 
			old_precision = precision
		old_recall = recall
	return score,total_recall

def miou(lst,ground):
	"""
	calculate mIoU through all the predictions
	"""
	cos_map,count_map,positive = match(lst,0,ground)
	miou = 0
	count = len(lst)
	real_count = 0
	for x in range(count):
		if cos_map[x]!=-1:
			miou += overlap(lst[x],ground[cos_map[x]])
			real_count += 1
	return miou/float(real_count) if real_count!=0 else 0.

def miou_per_v(lst,ground):
	"""
	calculate mIoU through all the predictions in one video first, then average the obtained mIoUs through single video.
	"""
	cos_map,count_map,positive = match(lst,0,ground)
	count = len(lst)
	v_miou = {}
	for x in range(count):
		if cos_map[x]!=-1:
			v_id = lst[x][4]
			miou = overlap(lst[x],ground[cos_map[x]])
			if v_id not in v_miou:
				v_miou[v_id] = [0.,0]
			v_miou[v_id][0] += miou
			v_miou[v_id][1] += 1
	miou = 0
	for v in v_miou:
		miou += v_miou[v][0]/float(v_miou[v][1])
	miou /= len(v_miou)
	return miou
