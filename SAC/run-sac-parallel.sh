#!/bin/bash

nohup python3 sac-fm_training.py --env_name 2x3_later_change --num_skills 3 --sparse > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x2 --num_skills 5 --sparse  > /dev/null 2>&1 &