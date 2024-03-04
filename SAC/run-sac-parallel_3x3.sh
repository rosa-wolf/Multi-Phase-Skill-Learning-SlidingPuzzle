#!/bin/bash

nohup python3 sac-fm_training.py --num_epochs 5000 --num_episodes 15 --env_name 3x3 --num_skills 4 --seed 123456 --prior_buffer > /dev/null 2>&1 &