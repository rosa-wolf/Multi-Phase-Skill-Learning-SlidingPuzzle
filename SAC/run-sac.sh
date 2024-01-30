#!/bin/bash

nohup  python3 sac-fm_training.py --env_name parallel_3x3_newcondition_logging --num_skills 4 > /dev/null 2>&1 &
nohup  python3 sac-fm_training.py --env_name parallel_3x3_newcondition_logging_relabeling08 --num_skills 4 --relabeling > /dev/null 2>&1 &
