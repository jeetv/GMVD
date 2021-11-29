# Bringing Generalization to Deep Multi-view Detection
<img src="./extras/three_generalization.png" height="400">

## Abstract
Multi-view Detection (MVD) is highly effective for oc-clusion reasoning in a crowded environment. While recentworks using deep learning have made significant advancesin the field, they have overlooked the generalization aspect,which makes themimpractical for real-world deployment.The key novelty of our work is toformalizethree criticalforms of generalization andpropose experiments to evaluatethem: generalization with i) a varying number of cameras, ii)varying camera positions, and finally, iii) to new scenes. Wefind that existing state-of-the-art models show poor general-ization by overfitting to a single scene and camera configu-ration. To address the concerns: (a) we propose a novel Gen-eralized MVD (GMVD) dataset, assimilating diverse sceneswith changing daytime, camera configurations, varying num-ber of cameras, and (b) we discuss the properties essentialto bring generalization to MVD and propose a barebonesmodel to incorporate them. We perform a comprehensive setof experiments on the WildTrack, MultiViewX and the GMVDdatasets to motivate the necessity to evaluate generalizationabilities of MVD methods and to demonstrate the efficacy ofthe proposed approach.

## Architecture
<img src="./extras/gmvd_arch.png" height="400">

## GMVD Dataset
<img src="./extras/gmvd_dataset.png" height="300" width="1000">
* Dataset links and instructions are provided here [link](https://github.com/jeetv/GMVD_dataset)

## Publicly Available Dataset
* Wildtrack dataset can be downloaded from this [link](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/).
* MultiviewX dataset can be downloaded from this [link](https://github.com/hou-yz/MultiviewX).

## Installation and Dependencies

* Install all requirements required to run the code on a Python 3.x by:
 ```	
# First, activate a new virtual environment
$ pip install -r requirements.txt
 ```
* Installation using conda :
 ```
$ conda env create -f environment.yml
 ```
 
* Download dataset and place it inside folder `GMVD/`
* Copy config.json file to Dataset folder 
```
# For Wildtrack
[GMVD]$ cp configuration/wildtrack/config.json ~/GMVD/Wildtrack/

# For MultiviewX
[GMVD]$ cp configuration/multiviewx/config.json ~/GMVD/MultiviewX/
```

## General Intructions
* All the experiments are perfomed using 1 Nvidia 1080Ti GPU's

## Training
For training, 
* ``training_commands/train.sh`` contains commands to run training in normal setting.
* ``training_commands/train_dropview.sh`` contains commands to run training with dropview regularization.

## Inference
* Clone this repository and download the pretrained weights from this [link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/jeet_vora_research_iiit_ac_in/EoZySkQaB2NAuBqbyGwwwX0BP4Ma33QIWdMvlJrczeQoHQ?e=2Z7xgT)
* Arguments to specific
```
--avgpool : to use average pooling
--dropview : enable dropview (note: --avgpool is also activated)
-d <dataset_name> : specify dataset eg:- wildtrack/multiviewx
```

* Inference for varying cameras
```
## Syntax Example : python main.py --avgpool --cam_set --train_cam 1 2 3 4 5 6 7 --test_cam 1 2 3 4 --resume <foldername>/<filename.pth>

# For Wildtrack
[GMVD]$ python main.py --avgpool --cam_set --train_cam 1 2 3 4 5 6 7 --test_cam 1 2 3 4 --resume trained_models/wildtrack/traditional_eval/Multiview_Detection_wildtrack.pth

# For MultiviewX
[GMVD]$ python main.py -d multiviewx --avgpool --cam_set --train_cam 1 2 3 4 5 6 --test_cam 1 2 3 4 --resume trained_models/multiviewx/traditional_eval/Multiview_Detection_multiviewx.pth
```

* Inference for changing camera configurations
```
## Syntax Example : python main.py --avgpool --cam_set --train_cam 2 4 5 6 --test_cam 1 3 5 7 --resume <foldername>/<filename.pth>

# For Wildtrack
[GMVD]$ python main.py --avgpool --cam_set --train_cam 2 4 5 6 --test_cam 1 3 5 7 --resume trained_models/wildtrack/changing_cam/Multiview_Detection_wildtrack_2456.pth

# For MultiviewX
[GMVD]$ python main.py -d multiviewx --avgpool --cam_set --train_cam 1 3 4 --test_cam 2 5 6 --resume trained_models/multiviewx/changing_cam/Multiview_Detection_multiviewx_134.pth
```

* Inference for scene generalization
```
[GMVD]$ python main.py -d wildtrack --avgpool --resume trained_models/multiviewx/traditional_eval/Multiview_Detection_multiviewx.pth
```

## Citations
```
@misc{vora2021bringing,
      title={Bringing Generalization to Deep Multi-view Detection}, 
      author={Jeet Vora and Swetanjal Dutta and Shyamgopal Karthik and Vineet Gandhi},
      year={2021},
      eprint={2109.12227},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
If you use the above code also cite this:
```
@inproceedings{hou2020multiview,
  title={Multiview Detection with Feature Perspective Transformation},
  author={Hou, Yunzhong and Zheng, Liang and Gould, Stephen},
  booktitle={ECCV},
  year={2020}
}

```
