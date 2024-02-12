#!/bin/bash

nohup python3 sac-fm_training.py --env_name 2x3-new-penalty --num_skills 3 --relabeling > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x3-new-penalty --num_skills 3 --sparse --relabeling > /dev/null 2>&1 &