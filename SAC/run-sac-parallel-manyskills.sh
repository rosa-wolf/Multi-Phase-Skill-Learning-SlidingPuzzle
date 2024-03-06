#!/bin/bash

nohup python3 sac-fm_training.py --env_name 2x2-2skills --num_epochs 5000 --seed 123456 --num_skills 2 --num_episodes 20 --num_steps 100 --sparse > log.txt 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x2-5skills --num_epochs 5000 --seed 123456 --num_skills 5 --num_episodes 20 --num_steps 100 --sparse > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x2-7skills --num_epochs 5000 --seed 123456 --num_skills 7 --num_episodes 20 --num_steps 100 --sparse > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x2-10skills --num_epochs 5000 --seed 123456 --num_skills 10 --num_episodes 20 --num_steps 100 --sparse > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x2-12skills --num_epochs 5000 --seed 123456 --num_skills 12 --num_episodes 20 --num_steps 100 --sparse > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x2-14skills --num_epochs 5000 --seed 123456 --num_skills 14 --num_episodes 20 --num_steps 100 --sparse > /dev/null 2>&1 &