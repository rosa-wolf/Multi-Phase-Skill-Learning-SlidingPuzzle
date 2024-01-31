#!/bin/bash

nohup  python3 sac-training-with-trained-fm.py  --env_name parallel_3x3_neg_dist_with-pretrained-fm --num_skills 4 > /dev/null 2>&1 &
