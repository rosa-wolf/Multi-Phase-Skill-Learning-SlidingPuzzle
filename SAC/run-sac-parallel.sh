#!/bin/bash

nohup python3 sac-fm_training.py --env_name 3x3 --num_skills 4 > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x3 --num_skills 4 --sparse --relabeling > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x3 --num_skills 4 --sparse > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x2 --sparse --num_skills 8 > /dev/null 2>&1 &
