#!/bin/bash

nohup python3 sac-fm_training.py  --num_epochs 20000 --env_name parallel_2x2 --reward_on_end  --num_skills 2 > /dev/null 2>&1 &
nohup python3 sac-train.py  --num_epochs 50000 --env_name skill_conditioned_3x3 --neg_dist --movement --reward_on_change --num_skills 4 > /dev/null 2>&1 &
nohup  python3 sac-train.py --env_name skill_conditioned_3x3 --num_skills 1 --reward_on_change --sparse > /dev/null 2>&1 &
nohup  python3 sac-train.py --env_name skill_conditioned_3x3 --num_skills 1 --reward_on_change --neg_dist_reward --movement_reward > /dev/null 2>&1 &