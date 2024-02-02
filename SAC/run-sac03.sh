#!/bin/bash

nohup python3 sac-fm_training.py --env_name parallel_3x3_negdist_entropy45 --relabeling --num_skills 4 > /dev/null 2>&1 &
nohup python3 sac-training-with-trained-fm.py --env_name parallel_3x3_negdist_entropy45 --relabeling --num_skills 4 > /dev/null 2>&1 &