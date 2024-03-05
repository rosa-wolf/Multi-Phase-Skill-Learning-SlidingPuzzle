#!/bin/bash

nohup python3 SEADS-train.py --env_name seads_3x3 --num_epochs 20000 --seed 123456 --num_skills 2 --num_episodes 20 --num_steps 200 --novelty_bonus --second_best > /dev/null 2>&1 &