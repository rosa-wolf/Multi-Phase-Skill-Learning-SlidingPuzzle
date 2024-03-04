#!/bin/bash

nohup python3 sac-fm_training.py --num_epochs 20000 --num_episodes 15 --env_name 3x3_constant --num_skills 4 --seed 123456 --prior_buffer > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --num_epochs 10000 --num_episodes 15 --env_name 2x3_constant --num_skills 3 --seed 123456 --prior_buffer > /dev/null 2>&1 &