#!/bin/bash

nohup python3 sac-train-new.py --env_name 2x2_2skills_predefined --num_epochs 2000 --dict_obs --seed 487193 --neg_dist_reward > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x2_2skills_predefined --num_epochs 2000 --dict_obs --seed 978623 --neg_dist_reward > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x2_2skills_predefined --num_epochs 2000 --dict_obs --seed 682147 --neg_dist_reward > /dev/null 2>&1 &