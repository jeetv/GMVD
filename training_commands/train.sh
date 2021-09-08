#!/bin/bash 
##################### Wildtrack Dataset training #####################
## Traditional training
python main.py --avgpool --earlystop 8

## Changing Camera Config
python main.py --avgpool --cam_set --train_cam 2 4 5 6 --test_cam 2 4 5 6 --earlystop 8
python main.py --avgpool --cam_set --train_cam 1 3 5 7 --test_cam 1 3 5 7 --earlystop 7

##################### MultiviewX Dataset training #####################
## Traditional training
python main.py -d multiviewx --avgpool --earlystop 9

## Changing Camera Config
python main.py -d multiviewx --avgpool --cam_set --train_cam 1 3 4 --test_cam 1 3 4 --earlystop 8
python main.py -d multiviewx --avgpool --cam_set --train_cam 2 5 6 --test_cam 2 5 6 --earlystop 8
