# Task-Consistency for R-C3D

### Test Environment

* Operating system - Ubuntu 16.04
* Language - Python 3.5.2
* Several dependencies -
  - numpy 1.15.3
  - terminaltables 3.1.0

### Result Refinement

[1] Refine the scores:

```sh
python3 result_refine.py <results> <info1> <info_c3d>
```

`<results>` is the score file in JSON format outputted by R-C3D. `<info1>` is the canonical database file of dataset COIN in JSON format which can be downloaded from the [website of COIN](...). `<info_c3d>` is the database file of dataset required by R-C3D. 

JSON `<info1>` is required to have the structure like:

```
{
	"database": {
		<video_id, str>: {
			"video_url": <video_url, str>,
			"duration": <video_duration, float>,
			"recipe_type": <target_id, int>,
			"class": <target_class, str>,
			"subset": ("training"|"validation"),
			"start": <random point between the start of the whole video and the start of the first action, float>,
			"end": <random point between the end of the whole video and the end of the last action, float>
			"annotation": [
				{
					"id": <action_id, int>,
					"segment": [start, end],
					"label": <action_label, str>
				},
				...
			]
		},
		...
	}
}
```

JSON `<info_c3d>` is required to have the structure like:

```
{
	"version": <version, str>,
	"taxonomy": [
		{
			"parentID": <id of the parent node, int>,
			"parentName": <name of parent node, str>, //There is supposed to be a global root node with name of "Root"
			"nodeID": <id of this node, int>,
			"nodeName": <name of this node, str>
		},
		...
	],
	"database": {
		<video_id, str>: {
			"video_url": <video_url, str>,
			"duration": <video_duration, float>,
			"resolution": "<width>x<height>",
			"subset": ("training"|"validation"),
			"annotation": [
				{
					"label": <action_id, int>,
					"segment": [start, end],
				},
				...
			]
		},
		...
	}
```

The refined scores will be dumped into a new JSON file with name of `<results>` suffixed with `.new`. 

[2] Calculate the metrics of refined results:

use `json_eval.py` to calculate the metrics.

```sh
python3 json_eval.py <info_c3d> <results>
```

`<info_c3d>` denotes the same meaning as in the first command. `<results>` is the refined result file with extension name as `result.json.new` if it hasn't been renamed. The `evaluate.py` module is required to launch this program. 

The module `evaluate.py` is forked from <https://github.com/ECHO960/PKU-MMD> and several functions we need in these programs are added.