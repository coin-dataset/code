## Benchmark Experiments
In order to provide a benchmark for our COIN dataset, we evaluate various of approaches under two different settings: step localization and action segmentation. We also conduct experiments on our task-consistency method under the first setting. The following provides the links of source codes. Thank the authors for sharing their code!

### Step Localization
In this task, we aim to localize a series of steps and recognize their corresponding labels given an instruction video. The following methods are evaluated:
* [SSN](https://github.com/yjxiong/action-detection) [1]
* [R-C3D](https://github.com/VisionLearningGroup/R-C3D) [2]
* Our Task Consistency Approach. Please see [tc-rc3d](tc-rc3d) and [tc-ssn](tc-ssn) for details.

### Action Segmentation
The goal of this task is to assign each video frame with a step label. The following methods are evaluated:
* [Action Sets](https://github.com/alexanderrichard/action-sets) [3]
* [NeuralNetwork-Viterbi](https://github.com/alexanderrichard/NeuralNetwork-Viterbi) [4]
* [TCFPN-ISBA](https://github.com/Zephyr-D/TCFPN-ISBA) [5]

Note that, these methods use frame-wise fisher vector as video representation, which comes with huge computation and storage cost on the COIN dataset (the calculation of fisher vector is based on the improved Dense Trajectory (iDT) representation, which requires huge computation cost and storage space). To address this, we employed a bidirectional LSTM on the top of a VGG16 network to extract dynamic feature of a video sequence[6].

### References
[1] Y. Zhao, Y. Xiong, L. Wang, Z. Wu, X. Tang, and D. Lin. Temporal action detection with structured segment networks. In ICCV, pages 2933–2942, 2017.
[2] H. Xu, A. Das, and K. Saenko. R-C3D: region convolutional 3d network for temporal activity detection. In ICCV, pages 5794–5803, 2017.
[3] A. Richard, H. Kuehne, and J. Gall. Action sets: Weakly supervised action segmentation without ordering constraints. In CVPR, pages 5987–5996, 2018.
[4] A. Richard, H. Kuehne, A. Iqbal, and J. Gall. Neuralnetwork-viterbi: A framework for weakly supervised video learning. In CVPR, pages 7386–7395, 2018.
[5] L. Ding and C. Xu. Weakly-supervised action segmentation with iterative soft boundary assignment. In CVPR, pages 6508–6516, 2018.
[6] J. Donahue, L. A. Hendricks, M. Rohrbach, S. Venugopalan, S. Guadarrama, K. Saenko, and T. Darrell. Long-term recurrent convolutional networks for visual recognition and description. TPAMI, 39(4):677–691, 2017.