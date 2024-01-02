#!/bin/bash

nohup python3 sac-fm_training.py  --num_epochs 20000 --env_name fm-policy-parallel > /dev/null 2>&1 &
nohup python3 sac-fm_training.py  --num_epochs 20000 --env_name fm-policy-parallel-r-on-end --reward_on_end > /dev/null 2>&1 &
nohup python3 sac-train.py  --num_epochs 2000 --reward_on_change --sparse > /dev/null 2>&1 &
nohup python3 sac-train.py  --num_epochs 2000 --reward_on_change > /dev/null 2>&1 &
nohup python3 sac-train.py  --num_epochs 1000 --env_name skill_conditioned_1x2 --reward_on_change --sparse > /dev/null 2>&1 &
nohup python3 sac-train.py  --num_epochs 50000 --env_name skill_conditioned_3x3 > /dev/null 2>&1 &