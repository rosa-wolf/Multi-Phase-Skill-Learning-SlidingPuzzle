#!/bin/bash


nohup python3 SEADS-train.py --env_name seads_2x2-15skills --num_epochs 5000 --seed 123456 --num_skills 15 --num_episodes 20 --num_steps 100 --novelty_bonus --second_best > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x2-20skills --num_epochs 5000 --seed 123456 --num_skills 15 --num_episodes 20 --num_steps 100 --novelty_bonus --second_best > /dev/null 2>&1 &