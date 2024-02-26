#!/bin/bash


nohup python3 sac-train-new.py --env_name 3x3_4skills_possible --num_epochs 2000 --dict_obs --seed 123456 --neg_dist_reward > /dev/null 2>&1 &