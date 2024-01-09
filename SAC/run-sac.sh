#!/bin/bash

nohup python3 sac-fm_training.py  --num_epochs 20000 --env_name parallel_1x2 --reward_on_end > /dev/null 2>&1 &
nohup python3 sac-fm_training.py  --num_epochs 20000 --env_name parallel_2x2 --reward_on_end  --num_skills 8 > /dev/null 2>&1 &
nohup python3 sac-train.py  --num_epochs 50000 --env_name skill_conditioned_3x3 --neg_dist --movement --reward_on_change > /dev/null 2>&1 &
nohup python3 sac-train.py  --num_epochs 50000 --env_name skill_conditioned_3x3 --neg_dist > /dev/null 2>&1 &
nohup python3 sac-train.py  --num_epochs 50000 --env_name skill_conditioned_3x3 --neg_dist --reward_on_change> /dev/null 2>&1 &
nohup python3 sac-train.py  --num_epochs 50000 --env_name skill_conditioned_3x3 --sparse --reward_on_change> /dev/null 2>&1 &