#!/bin/bash

nohup python3 sac-train.py --env_name 1x2 --num_epochs 35000 --reward_on_change --neg_dist_reward --seed 12345 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name 2x2 --num_epochs 35000 --reward_on_change --neg_dist_reward --seed 12345 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name 2x3 --num_epochs 35000 --reward_on_change --neg_dist_reward --seed 12345 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name 3x3 --num_epochs 35000 --reward_on_change --neg_dist_reward --seed 12345 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name 1x2 --num_epochs 35000 --reward_on_change --neg_dist_reward --seed 19471 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name 2x2 --num_epochs 35000 --reward_on_change --neg_dist_reward --seed 19471 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name 2x3 --num_epochs 35000 --reward_on_change --neg_dist_reward --seed 19471 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name 3x3 --num_epochs 35000 --reward_on_change --neg_dist_reward --seed 19471 > /dev/null 2>&1 &
