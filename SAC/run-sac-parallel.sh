#!/bin/bash

nohup python3 sac-fm_training.py --num_epochs 10000 --env_name 2x3-prior --num_skills 3 --prior_buffer --sparse > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --num_epochs 10000 --env_name 2x3 --num_skills 3 --sparse  > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --num_epochs 10000 --env_name 2x3-relabel --num_skills 3 --sparse > /dev/null 2>&1 &