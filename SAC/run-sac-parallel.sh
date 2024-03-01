#!/bin/bash

nohup python3 sac-fm_training.py --num_epochs 5000 --num_episodes 5 --env_name 2x2-sbnorm --num_skills 2  --sparse -seed 123456 --second_best > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --num_epochs 5000 --num_episodes 5 --env_name 2x2-sbnorm --num_skills 3  --sparse -seed 123456 --second_best > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --num_epochs 5000 --num_episodes 5 --env_name 2x2-sbnorm --num_skills 4  --sparse -seed 123456 --second_best > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --num_epochs 5000 --num_episodes 5 --env_name 2x2-sbnorm --num_skills 5  --sparse -seed 123456 --second_best > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --num_epochs 5000 --num_episodes 5 --env_name 2x2-sbnorm --num_skills 6  --sparse -seed 123456 --second_best > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --num_epochs 5000 --num_episodes 5 --env_name 2x2-sbnorm --num_skills 7  --sparse -seed 123456 --second_best > /dev/null 2>&1 &