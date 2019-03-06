# Task-Consistency for SSN

### Test Environment

* Operating system - Ubuntu 16.04
* Language - Python 3.5.2
* Several dependencies -
  - numpy 1.15.3
  - terminaltables 3.1.0
  - pandas 0.23.4

### The Structure of SSN Score File

The score file dumped by SSN is in format of `pkl`. It is serialised from a python `dict` in which the paths of video frames serve as keys and a 4-element tuple of numpy arrays serve as values. The meaning of four arrays is described as following:

* The shape of the 1st array in the tuple is (N,2) where N denotes the proposal number. The elements in this array indicates the lower and higher bounds of the proposal ranges.
* The shape of the 2nd array in the tuple is (N,K+1) where K denotes the number of action classes. There are the actionness scores in this array.
* The shape of the 3rd array in the tuple is (N,K). There are the completeness scores presented by SSN in this array.
* The shape of the 4th array in the tuple is (N,K,2). There are the regression scores in this array. The regression score is given as a 2-element array \[`center_regression`, `duration_regression`\]. The regression operation could be formularised as:

```
regressed_center = range_renter+range_duration*center_regression
regressed_duration = range_duration*exp(duration_regression)
```

### Get Combined Score File

The standalone score file of combined scores is required while refining the combined scores of RGB and Flow modality. The program derived from the original evaluation program is used to export the combined scores to a standalone `pkl` file. These programs are `fusion_pkl_generation_eval_detection_results.py` and `fusion_eval_detection_results.py`. Either the program exports the same `pkl` file.

While launching the program, use `--dump_combined` option to indicate the output file. If not set, the scores will be dumped into `ssn_fusion.pkl` by default.

```sh
python3 fusion_pkl_generation_eval_detection_results.py coin_small <RGB_score> <Flow_score> --score_weights 2 1 --dump_combined <dump_file>
```

### Result Refinement

[1] Use `data_processing.py` to process the `pkl`-format score file and calculate the scores of actionness and completeness and dump to several numpy files.

```sh
python3 data_processing.py <pkl_score> <json_info>
```

`<pkl_score>` is the `pkl` score file to process. `<json_info>` is JSON-format database of the dataset. About the structure of this file, please refer to [TC for R-C3D](../tc-c3d/README.md). The generated `npy` files are saved under the directory with name which is the same as the main file name of `<pkl_score>`.

[2] Use `gen_matrix.py` to generate the constraints matrix denoting the consistency among action classes and target classes. 

```sh
python3 gen_matrix.py <json_file> <npy_file>
```

`<json_file>` is the database of the dataset as mentioned above. `<npy_file>` is the output matrix.

[3] Use `combined_refine.py` to refine the scores.

```sh
python3 combined_refine.py -c <npy_constrains> -i <src_scores> -o <refined_scores>
```

`<npy_constrains>` is the constraints matrix mentioned above. `<src_scores>` is the directory of the numpy-format scores mentioned in "1.". `<refined_scores>` is the output directory of the refined scores.

[4] Calculate the metrics of the refined scores. Use the program derived from the original evaluation program of SSN, `combined_eval_detection_results.py` to evaluate the refined scores. Please set the `--externel_score` option to import the refined scores from the corresponding directory, or the program will attempt to import the scores from `test_gt_score_combined_refined_fusion`. And the original unrefined `pkl`-format score file is also required to extract the regression scores which have not been adjusted.

```sh
python3 combined_eval_detection_results.py coin_small <combined_score> --externel_score <external_score>
```
