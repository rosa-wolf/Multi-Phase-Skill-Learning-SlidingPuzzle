#!/bin/bash

nohup python3 sac-fm_training.py --env_name 2x3-prior --num_skills 3 --prior_buffer > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x3 --num_skills 3  > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x3-relabel --num_skills 3 --relabeling > /dev/null 2>&1 &