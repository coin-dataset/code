#!/usr/bin/python3

"""
Perform Ordering-Dependency (OD) and Task-Consistency (TC) method.

Composed by Danyang Zhang @THU_IVG
Last revision: Danyang Zhang @THU_IVG @Oct 3rd, 2019 CST
"""

import argparse
import os.path
import pickle as pkl
import json
import scipy.io as sio

import numpy as np
import itertools
import concurrent.futures as con

import terminaltables
import matplotlib.pyplot as plt

def _softmax(scores):
    exp_scores = np.exp(scores)
    softmax_scores = exp_scores/np.sum(exp_scores, axis=-1)[:, None]
    return softmax_scores

_simple_iou = lambda r1, r2: max(0, min(r1[1], r2[1])-max(r1[0], r2[0])) /\
        (max(r1[1], r2[1])-min(r1[0], r2[0]))
_piou = lambda p1, p2: max(0, min(p1[2], p2[2])-max(p1[1], p2[1])) /\
        (max(p1[2], p2[2])-min(p1[1], p2[1]))
_gpiou = lambda g, p: 0. if g[0]!=p[0] else _piou(g, p)

def _nms(video_prediction_info, thrd):
    preserved_proposals = []
    for p in video_prediction_info:
        if all(_gpiou(p, pp)<=thrd for pp in preserved_proposals):
            preserved_proposals.append(p)
    preserved_proposals = np.array(preserved_proposals)
    return preserved_proposals

def construct_actionness_distribution(combined_score, interval, nb_slot, density_function):
    nb_proposal, nb_action = combined_score.shape

    proposal_score_along_time_axis = np.zeros((nb_proposal, nb_slot+1, nb_action))
    independent_variables = np.arange(0, nb_slot+1)

    #reduced_score = np.sum(combined_score, axis=1)
    interval_center = (interval[:, 0] + interval[:, 1])/2.
    interval_duration = (interval[:, 1] - interval[:, 0])/2.
    interval_center *= float(nb_slot)
    interval_duration *= float(nb_slot)
    for i in range(nb_proposal):
        proposal_score_along_time_axis[i, :] = combined_score[i][None, :]*density_function(independent_variables, interval_center[i], interval_duration[i])[:, None]
    dist_along_time_axis = np.sum(proposal_score_along_time_axis, axis=0)

    return proposal_score_along_time_axis, dist_along_time_axis

def watershed_method(reduced_distribution, comparator, termination_indicator, indicator_thrd):
    max_score = np.amax(reduced_distribution)
    for act_thrd in np.arange(0.95, 0, -0.05):
        peak_detection = reduced_distribution>=max_score*act_thrd
        if comparator(termination_indicator(peak_detection), indicator_thrd):
            break
    return peak_detection

if __name__=="__main__":
    # load scores and merge them
    # load groundtruth
    # calculate combined scores
    # refine combined scores
    # get topk
    # perform NMS
    # perform regression onto the time range
    # traverse the different iou thresholds
    # perform match
    # calculate metrics (mAP, mAR, F1)

    parser = argparse.ArgumentParser()

    parser.add_argument("--matrix", type=str, required=True, help="consistency constraints matrix (numpy matrix)")
    parser.add_argument("--scores", nargs="+", type=str, required=True, help="output of SSN (pkl-format score file)")
    parser.add_argument("--weights", nargs="+", type=float, required=False, help="custom weights of the different score files")
    parser.add_argument("--groundtruth", type=str, required=True, help="groundtruth of COIN")

    parser.add_argument("--refinement", nargs="+", type=str, choices=["TC", "OD"], required=False, help="the refinement expected to be applied")

    parser.add_argument("--attenuation-coefficient", default=np.exp(-2.), type=float, required=False, help="attenuation coefficient for TC, default to exp(-2)")

    parser.add_argument("--markov-matrix", type=str, help="markov matrix for the markov method")
    parser.add_argument("--refine-head-proposal", action="store_true", required=False, help="refine the head proposal as well")
    parser.add_argument("--refinement-weights", nargs=2, type=float, required=False, help="refinement weights")

    parser.add_argument("--density-function", default="gaussian", type=str, choices=["uniform", "gaussian"], required=False, help="density function to use")
    parser.add_argument("--min-background-gap", default=6, type=int, required=False, help="minimum background gap")
    #parser.add_argument("--max-positive-length", default=10, type=int, required=False, help="maximum positive range length")
    #parser.add_argument("--max-positive-proportion", default=0.40, type=float, required=False, help="maximum positive frame proportion")

    parser.add_argument("--topk", type=int, default=60, required=False, help="only top k prediections in one video will be preserved")
    parser.add_argument("--nms", type=float, default=0.6, required=False, help="NMS threshold")
    parser.add_argument("--no-extra-background", action="store_true", required=False, help="if specified, no extra background column will be attracted to the consistency matrix, which means that the background action category has already been included into the consistency matrix")
    parser.add_argument("--nb-thread", type=int, default=32, required=False, help="the number of thread to use")
    args = parser.parse_args()

    # load consistency matrix
    if os.path.exists(args.matrix):
        consistency_matrix = np.load(args.matrix)
        if not args.no_extra_background:
            consistency_matrix = np.concatenate([np.zeros((consistency_matrix.shape[0], 1)), consistency_matrix],
                axis=1)
    else:
        print("Consistency matrix doesn't exist.")
        exit(1)

    # load scores and merge them
    if args.weights is not None:
        if len(args.weights)<len(args.scores):
            print("Please provide at least as much weight arguments as score files or ignore this option completely.")
            exit(2)
        weights = np.array(args.weights[:len(args.scores)])
    else:
        weights = np.ones((len(args.scores),))
    weight_sum = np.sum(weights)
    if weight_sum==0:
        print("Please confirm that the sum of the weights doesn't equal 0")
        exit(3)

    scores = []
    for fn in args.scores:
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                scores.append(pkl.load(f))
        else:
            print("Score file {:} not found.".format(fn))
            exit(4)

    ref_score_dict = scores[0]
    vids = ref_score_dict.keys()
    score_dict = {
            k: (ref_score_dict[k][0],
                sum(w*s[k][1] for w, s in zip(weights, scores)) / weight_sum,
                sum(w*s[k][2] for w, s in zip(weights, scores)) / weight_sum,
                sum(w*s[k][3] for w, s in zip(weights, scores)) / weight_sum)
            for k in vids
    }

    print("Score loaded and merged")

    # load groundtruth
    if os.path.exists(args.groundtruth):
        with open(args.groundtruth, "r") as f:
            database = json.load(f)["database"]
    else:
        print("Groundtruth database not found.")
        exit(5)

    nb_groundtruth_by_action_class = [0] * consistency_matrix.shape[1]
    groundtruth_dict = {}
    for v in database:
        if database[v]["subset"] != "training":
            groundtruth_dict[v] = [(int(s["id"]),
                float(s["segment"][0])/database[v]["duration"],
                float(s["segment"][1])/database[v]["duration"]) for s in database[v]["annotation"]]
            for g in groundtruth_dict[v]:
                nb_groundtruth_by_action_class[g[0]] += 1

    print("Groundtruth loaded")

    if args.refinement is not None and "OD" in args.refinement:
        # load the markov matrix
        if args.markov_matrix is None:
            print("Please provide the Markov matrix")
            exit(6)
        if os.path.exists(args.markov_matrix):
            markov_dict = sio.loadmat(args.markov_matrix)
        else:
            print("The given matrix file doesn't exist")
            exit(1)

        markov_weights = args.refinement_weights if args.refinement_weights is not None else [1., 1.]
        markov_weights_sum = sum(markov_weights)
        markov_weights = [w/markov_weights_sum for w in markov_weights]

    def perform_tc(combined_score, task_inference):
        combined_score = np.copy(combined_score)
        # refine the scores of corresponding actions
        refining_mask = np.full(combined_score.shape, args.attenuation_coefficient)
        refining_mask[:, 0] = 1.
        preserved_actions = np.where(consistency_matrix[task_inference])[0]
        refining_mask[:, preserved_actions] = 1.
        combined_score *= refining_mask
        return combined_score

    def perform_clustering_refinement(combined_score, interval):
        dist_funcs = {
                "gaussian": lambda x, mean, stdvar: np.exp(-(x-mean)**2/(2*stdvar**2))/(np.sqrt(2*np.pi)*stdvar), # normal distribution, probability density function
                "uniform": lambda x, center, span: np.where(np.absolute(x-center)<=span, 0.5/span, 0.)
        }
        combined_score  = np.copy(combined_score)

        # construct distribution along time axis
        nb_proposal, nb_action = combined_score.shape

        proposal_score_along_time_axis, dist_along_time_axis = construct_actionness_distribution(combined_score, interval, 100, dist_funcs[args.density_function])

        # split into proposals
        def check_max_length(peak_detection):
            groups = itertools.groupby(peak_detection)
            lengths = [len(list(g)) for k, g in groups if k]
            return max(lengths) if len(lengths)>0 else 0.
        def check_min_length(peak_detection):
            groups = itertools.groupby(peak_detection)
            lengths = [len(list(g)) for k, g in groups if k]
            return min(lengths) if len(lengths)>0 else 0.
        def check_avg_length(peak_detection):
            groups = itertools.groupby(peak_detection)
            lengths = [len(list(g)) for k, g in groups if k]
            return sum(lengths)/float(len(lengths)) if len(lengths)>0 else 0.
        def check_max_gap(peak_detection):
            groups = itertools.groupby(peak_detection)
            lengths = [len(list(g)) for k, g in groups if not k]
            return max(lengths) if len(lengths)>0 else 0.
        def check_min_gap(peak_detection):
            groups = itertools.groupby(peak_detection)
            lengths = [len(list(g)) for k, g in groups if not k]
            return min(lengths) if len(lengths)>0 else 0.
        def check_avg_gap(peak_detection):
            groups = itertools.groupby(peak_detection)
            lengths = [len(list(g)) for k, g in groups if not k]
            return sum(lengths)/float(len(lengths)) if len(lengths)>0 else 0.
        check_positive_proportion = lambda p: np.sum(p)/float(len(p))
        #termination_indicator = check_avg_length
        #termination_indicator = check_max_length
        #termination_indicator = check_min_length
        #termination_indicator = check_max_gap
        #termination_indicator = check_min_gap
        termination_indicator = check_avg_gap
        #termination_indicator = check_positive_proportion
        #comparator = np.greater_equal
        comparator = np.less_equal
        #indicator_thrd = args.max_positive_length
        indicator_thrd = args.min_background_gap
        #indicator_thrd = args.max_positive_proportion

        reduced_distribution = np.sum(dist_along_time_axis, axis=1)
        peak_detection = watershed_method(reduced_distribution, comparator, termination_indicator, indicator_thrd)

        # refine the scores
        # determine the action ranges
        grouped_slots = itertools.groupby(enumerate(peak_detection), key=(lambda t: t[1]))
        grouped_slots = [list(s[0] for s in g) for k, g in grouped_slots if k]
        cluster_score = [np.sum(dist_along_time_axis[sl], axis=0) for sl in grouped_slots]
        cluster_score = np.array(cluster_score) if len(cluster_score)>0 else np.zeros((0, nb_action)) #shape: (nb_cluster, nb_action)

        # Markov refinement
        cluster_score_sum = np.sum(cluster_score, axis=1)
        normalized_cluster_score = cluster_score/cluster_score_sum[:, None]
        normalized_cluster_score = np.nan_to_num(normalized_cluster_score)

        # apply that refinement
        init_dist = np.squeeze(markov_dict["normalized_init_dist"]) if args.no_extra_background\
                else np.concatenate([[0.],
                    np.squeeze(markov_dict["normalized_init_dist"])])
        markov_matrix = markov_dict["normalized_frequency_mat"]
        if not args.no_extra_background:
            markov_matrix = np.concatenate([np.zeros((markov_matrix.shape[0], 1)), markov_matrix], axis=1)
            markov_matrix = np.concatenate([np.zeros((1, markov_matrix.shape[1])), markov_matrix], axis=0)
            markov_matrix[0][0] = 1.

        if args.refine_head_proposal and len(normalized_cluster_score)>=1:
            normalized_cluster_score[0] = markov_weights[0]*normalized_cluster_score[0] + markov_weights[1]*init_dist
        for i in range(1, len(normalized_cluster_score)):
            normalized_cluster_score[i] = markov_weights[0]*normalized_cluster_score[i] +\
                    markov_weights[1]*np.matmul(normalized_cluster_score[i-1], markov_matrix)

        refined_cluster_score = normalized_cluster_score*cluster_score_sum[:, None]

        # back-propagete the refined scores into each frames, and then into each proposals
        refining_coefficients = refined_cluster_score/cluster_score
        refined_distribution = np.copy(dist_along_time_axis)
        for rfc, sl in zip(refining_coefficients, grouped_slots):
            refined_distribution[sl] = dist_along_time_axis[sl]*rfc
        refining_shift = refined_distribution-dist_along_time_axis #shape: (nb_slot+1, nb_action)
        proposal_proportion = proposal_score_along_time_axis/dist_along_time_axis
        proposal_proportion = np.nan_to_num(proposal_proportion) #shape: (nb_proposal, nb_slot+1, nb_action)
        time_wise_proportion = proposal_score_along_time_axis/combined_score[:, None, :]
        time_wise_proportion = np.nan_to_num(time_wise_proportion)
        combined_score += np.sum(time_wise_proportion*proposal_proportion*refining_shift, axis=1)
        return combined_score

    def preprocess_predictions(vid):
        # calculate combined scores
        actionness = score_dict[vid][1][:, 1:]
        completeness = score_dict[vid][2]

        combined_score = _softmax(actionness)*np.exp(completeness)
        # shape: (nb_proposal, nb_action_category)

        interval = score_dict[vid][0]
        regression = score_dict[vid][3]

        # refine combined scores
        # infer the likely task of video

        if args.refinement is not None:
            if "TC" in args.refinement:
                video_score = np.sum(combined_score, axis=0)
                video_task_score = np.matmul(consistency_matrix, video_score)
                task_inference = np.argmax(video_task_score)
            for rfm in args.refinement:
                if rfm=="TC":
                    combined_score = perform_tc(combined_score, task_inference)
                elif rfm=="OD":
                    combined_score = perform_clustering_refinement(combined_score, interval)

        # get topk
        # add action id to proposal info and stack the different items
        nb_proposal, nb_action = combined_score.shape

        interval = np.reshape(interval, (-1, 1, 2))
        interval = np.broadcast_to(interval, regression.shape)

        action_ids = np.arange(nb_action)
        action_ids = np.broadcast_to(action_ids, combined_score.shape)
        action_ids = np.reshape(action_ids, (nb_proposal, nb_action, 1))

        combined_score = np.reshape(combined_score, (nb_proposal, nb_action, 1))
        video_prediction_info = np.concatenate([action_ids, interval, regression, combined_score], axis=-1)
        # sort and retrieve the top k
        sorted_indices = np.argsort(combined_score, axis=None)
        video_prediction_info = np.reshape(video_prediction_info, (-1, 6))
        video_prediction_info = video_prediction_info[sorted_indices[:-args.topk-1:-1]]

        # perform NMS
        preserved_proposals = _nms(video_prediction_info, args.nms)

        # perform regression onto the time range
        interval = preserved_proposals[:, 1:3]
        regression = preserved_proposals[:, 3:5]

        interval_center = (interval[:, 0] + interval[:, 1])/2.
        interval_duration = interval[:, 1] - interval[:, 0]

        interval_center = interval_center + interval_duration*regression[:, 0]
        interval_duration = interval_duration * np.exp(regression[:, 1])

        preserved_proposals[:, 1] = np.maximum(0., interval_center-interval_duration/2.)
        preserved_proposals[:, 2] = np.minimum(1., interval_center+interval_duration/2.)

        return preserved_proposals

    video_predictions = {}
    preprocessing_pool = con.ThreadPoolExecutor(args.nb_thread)
    preprocessing_futures = {}
    for v in score_dict:
        preprocessing_futures[v.split("/")[-1]] = preprocessing_pool.submit(preprocess_predictions, v)
    for v in preprocessing_futures:
        video_predictions[v] = preprocessing_futures[v].result()

    print("Prepare to calculate metrics")

    def cal_metrics(iou_thrd):
        # perform match
        match_result_by_action_class = [[] for _ in range(consistency_matrix.shape[1])]
        # elements in this list:
        # tuples with form like: (video_id as str, proposal as np.array([action_id, start, end, center_regression, duration_regression, score]), matched as bool)
        for v in video_predictions:
            groundtruth = groundtruth_dict[v]

            groundtruth_matched = [False] * len(groundtruth)
            nb_unmatched_groundtruth = len(groundtruth)
            for p in video_predictions[v]:
                if nb_unmatched_groundtruth==0:
                    match_result_by_action_class[int(p[0])].append((v, p, False))
                    continue

                max_iou = 0.
                best_matched_groundtruth_index = None
                for j, g in enumerate(groundtruth):
                    if not groundtruth_matched[j]:
                        cur_iou = _gpiou(g, p)
                        if cur_iou>iou_thrd and cur_iou>max_iou:
                            max_iou = cur_iou
                            best_matched_groundtruth_index = j

                if best_matched_groundtruth_index is not None:
                    match_result_by_action_class[int(p[0])].append((v, p, True))
                    groundtruth_matched[best_matched_groundtruth_index] = True
                    nb_unmatched_groundtruth -= 1
                else:
                    match_result_by_action_class[int(p[0])].append((v, p, False))

        # calculate metrics (mAP, mAR, F1)
        aps = []
        ars = []
        for i, ele in enumerate(zip(match_result_by_action_class, nb_groundtruth_by_action_class)):
            if i==0:
                continue
            m, nbg = ele
            if nbg==0:
                aps.append(0.)
                ars.append(1.)
                continue
            if len(m)==0:
                aps.append(0.)
                ars.append(0.)
                continue
            m.sort(key=(lambda p: p[1][5]), reverse=True)
            match_result = [p[2] for p in m]

            match_result = np.array(match_result, dtype=np.float64)
            recall_func = np.cumsum(match_result)
            precision_func = recall_func/np.arange(1, len(recall_func)+1, dtype=np.float64)
            recall_func /= float(nbg)
            match_result /= float(nbg)

            for j in range(len(precision_func)-2, -1, -1):
                precision_func[j] = max(precision_func[j+1], precision_func[j])
            aps.append(np.sum(match_result*precision_func))
            ars.append(recall_func[-1])

        return sum(aps)/float(len(aps)), sum(ars)/float(len(ars))

    # traverse the different iou thresholds
    maps = []
    mars = []
    cal_metrics_pool = con.ThreadPoolExecutor(args.nb_thread)
    cal_metrics_futures = []
    for thrd in np.arange(0.1, 1.0, 0.1):
        cal_metrics_futures.append(cal_metrics_pool.submit(cal_metrics, thrd))
    for f in cal_metrics_futures:
        _map, mar = f.result()
        maps.append(_map)
        mars.append(mar)

    table_data = [
            ["IoU Threshold"] + ["{:.2f}".format(i) for i in np.arange(0.1, 1.0, 0.1)],
            ["mAP"] + ["{:.4f}".format(p) for p in maps],
            ["mAR"] + ["{:.4f}".format(r) for r in mars]
    ]
    table = terminaltables.AsciiTable(table_data)
    table.inner_row_border = True
    print(table.table)
