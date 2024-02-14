#!/bin/bash

nohup python3 sac-fm_training.py --env_name 3x3 --num_skills 4 --prior_buffer > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 3x3 --num_skills 4 --prior_buffer --relabeling > /dev/null 2>&1 &
