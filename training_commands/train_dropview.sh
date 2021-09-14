#!/bin/bash 
##################### Wildtrack Dataset with DropView training #####################
## Traditional training
python main.py --dropview

## Changing Camera Config
python main.py --dropview --cam_set --train_cam 2 4 5 6 --test_cam 2 4 5 6
python main.py --dropview --cam_set --train_cam 1 3 5 7 --test_cam 1 3 5 7

##################### MultiviewX Dataset with DropView training #####################
## Traditional training
python main.py -d multiviewx --dropview

## Changing Camera Config
python main.py -d multiviewx --dropview --cam_set --train_cam 1 3 4
python main.py -d multiviewx --dropview --cam_set --train_cam 2 5 6
