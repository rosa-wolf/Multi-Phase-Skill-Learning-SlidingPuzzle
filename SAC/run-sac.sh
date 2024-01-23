#!/bin/bash

nohup  python3 sac-fm_training.py --env_name parallel_3x3 --num_skills 4 > /dev/null 2>&1 &
nohup  python3 sac-fm_training.py --env_name parallel_3x3 --num_skills 2 > /dev/null 2>&1 &
nohup  python3 sac-fm_training.py --env_name parallel_2x2 --num_skills 2 --relabeling > /dev/null 2>&1 &
