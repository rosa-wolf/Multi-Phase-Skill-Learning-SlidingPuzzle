#!/bin/bash

nohup python3 sac-fm_training.py --env_name 2x2-15skills --num_epochs 5000 --seed 123456 --num_skills 15 --num_episodes 20 --num_steps 100 --sparse --second_best > log.txt 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x2-20skills --num_epochs 5000 --seed 123456 --num_skills 20 --num_episodes 20 --num_steps 100 --sparse --second_best > /dev/null 2>&1 &