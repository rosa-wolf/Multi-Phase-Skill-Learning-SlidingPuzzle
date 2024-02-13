#!/bin/bash

nohup python3 sac-fm_training.py --env_name 3x3-new-penalty --num_skills 4 > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x2-new-penalty --num_skills 8 > /dev/null 2>&1 &
nohup python3 sac-training-with-trained-fm.py --env_name 3x3_pretrained_fm --num_skills 4 > /dev/null 2>&1 &
