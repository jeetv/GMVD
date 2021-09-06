# Bringing Generalization to Deep Multi-view Detection
<img src="./extras/three_generalization.png" height="400">

## Abstract
Multi-view Detection (MVD) is highly effective for occlusion reasoning and is a
mainstream solution in various applications that require accurate top-view occupancy
maps. While recent works using deep learning have made significant advances in the
field, they have overlooked the generalization aspect, which makes them impractical
for real-world deployment. The key novelty of our work is to formalize three critical
forms of generalization and propose experiments to investigate them: i) generalization
across a varying number of cameras, ii) generalization with varying camera positions,
and finally, iii) generalization to new scenes. We find that existing state-of-the-art models
show poor generalization by overfitting to a single scene and camera configuration. We
propose modifications in terms of pre-training, pooling strategy, regularization, and loss
function to an existing state-of-the-art framework, leading to successful generalization
across new camera configurations and new scenes. We perform a comprehensive set of
experiments on the WildTrack and MultiViewX datasets to (a) motivate the necessity to
evaluate MVD methods on generalization abilities and (b) demonstrate the efficacy of
the proposed approach.

## Architecture
<img src="./extras/MVDarch.png" height="400">

## Dataset
* Wildtrack Dataset can be downloaded from this [link](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/).
* MultiviewX Dataset can be downloaded from this [link](https://github.com/hou-yz/MultiviewX).

## Results
### Traditional Evaluation
![](./extras/traditional_eval.PNG)
### Varying Number of Cameras
![](./extras/vary_cam.png)
### Changing Camera Configurations
![](./extras/change_cam.PNG)
### Scene Generalization
![](./extras/sc_gen.PNG)
