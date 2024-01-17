#!/bin/bash

nohup  python3 sac-train.py --env_name skill_conditioned_3x3 --reward_on_change --neg_dist_reward --movement_reward --num_skills 1 > /dev/null 2>&1 &
nohup  python3 sac-train.py --env_name skill_conditioned_3x3 --reward_on_change --neg_dist_reward --movement_reward --num_skills 4 > /dev/null 2>&1 &
nohup  python3 sac-train.py --env_name skill_conditioned_3x3 --reward_on_change --neg_dist_reward --movement_reward --num_skills 6 > /dev/null 2>&1 &