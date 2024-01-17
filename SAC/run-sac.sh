#!/bin/bash

nohup  python3 sac-fm_training.py --num_epochs 20000 --env_name parallel_1x2 --num_skills 2 --num_episodes 2 --relabeling > /dev/null 2>&1 &
nohup  python3 sac-fm_training.py --num_epochs 20000 --env_name parallel_1x2 --num_skills 1 --num_episodes 2 --relabeling > /dev/null 2>&1 &
nohup  python3 sac-fm_training.py --num_epochs 20000 --env_name parallel_1x2 --num_skills 2 --num_episodes 2 > /dev/null 2>&1 &