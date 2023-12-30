#!/bin/bash

nohup python3 sac-fm_training.py  --num_epochs 10000 --env_name fm-policy-parallel > /dev/null 2>&1 &
nohup python3 sac-train.py  --num_epochs 2000 > /dev/null 2>&1 &
nohup python3 sac-train.py  --num_epochs 15000 --skill_conditioned_3x3 > /dev/null 2>&1 &