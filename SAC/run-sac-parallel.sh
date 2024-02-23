#!/bin/bash

nohup python3 sac-fm_training.py --env_name 2x3 --num_skills 3 > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x3 --num_skills 3 --sparse > /dev/null 2>&1 &