#!/bin/bash

nohup  python3 sac-fm_training.py --env_name 2x2 --num_skills 2 > /dev/null 2>&1 &
nohup  python3 sac-fm_training.py --env_name 2x2 --num_skills 2 --sparse > /dev/null 2>&1 &