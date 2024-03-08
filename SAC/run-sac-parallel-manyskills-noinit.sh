#!/bin/bash

nohup python3 sac-fm_training.py --env_name 2x2-15skills-noinit --num_epochs 5000 --seed 578907 --num_skills 15 --num_episodes 20 --num_steps 100 --sparse --second_best --dorefinement > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x2-15skills-noinit --num_epochs 5000 --seed 937584 --num_skills 15 --num_episodes 20 --num_steps 100 --sparse --second_best --dorefinement > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x2-15skills-noinit --num_epochs 5000 --seed 105399 --num_skills 15 --num_episodes 20 --num_steps 100 --sparse --second_best --dorefinement > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x2-15skills-noinit --num_epochs 5000 --seed 195738 --num_skills 15 --num_episodes 20 --num_steps 100 --sparse --second_best --dorefinement > /dev/null 2>&1 &