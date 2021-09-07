#!/bin/bash 
##################### Wildtrack Dataset with DropView training #####################
## Traditional training
python main.py --avgpool --dropview --earlystop 8

## Changing Camera Config
python main.py --avgpool --dropview --cam_set --train_cam 2 4 5 6 --test_cam 2 4 5 6
python main.py --avgpool --dropview --cam_set --train_cam 1 3 5 7 --test_cam 1 3 5 7 --earlystop 9

##################### MultiviewX Dataset eith DropView training #####################
## Traditional training
python main.py -d multiviewx --avgpool --dropview

## Changing Camera Config
python main.py -d multiviewx --avgpool --dropview --cam_set --train_cam 1 3 4 --test_cam 1 3 4
python main.py -d multiviewx --avgpool --dropview --cam_set --train_cam 2 5 6 --test_cam 2 5 6
