#!/bin/bash

nohup python3 sac-training-with-trained-fm.py --env_name parallel_3x3_relabeling_shaping_rewards --num_skills 4 --relabeling > /dev/null 2>&1 &