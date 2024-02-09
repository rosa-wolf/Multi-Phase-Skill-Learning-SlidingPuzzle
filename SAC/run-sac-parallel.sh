#!/bin/bash


nohup python3 sac-train-old.py --env_name skill_conditioned_2x2_no-coord --num_epochs 2000 --dict_obs --seed 123456 > /dev/null 2>&1 &