#!/bin/bash

nohup  python3 sac-fm_training.py --env_name parallel_3x3_newcondition --num_skills 4 > /dev/null 2>&1 &
