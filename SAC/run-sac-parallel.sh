#!/bin/bash

nohup python3 sac-fm_training.py --num_epochs 10000 --num_episodes 10 --gamma 0.93 --lr 0.0005 --env_name 2x3-prior --num_skills 3 --prior_buffer --sparse -seed 123456 > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --num_epochs 10000 --num_episodes 10 --gamma 0.93 --lr 0.0005 --env_name 2x3 --num_skills 3 --sparse --seed 123456  > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --num_epochs 10000 --num_episodes 10 --gamma 0.93 --lr 0.0005 --env_name 2x3-relabel --num_skills 3 --sparse --relabeling  --seed 123456 > /dev/null 2>&1 &