#!/bin/bash

nohup python3 sac-fm_training.py --env_name 2x3-newsampling-05-01 --num_epochs 7000 --seed 123456 --num_skills 10 --num_episodes 20 --num_steps 100 --sparse --second_best --doinit --dorefinement > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x3-newsampling-05-01 --num_epochs 7000 --seed 578907 --num_skills 10 --num_episodes 20 --num_steps 100 --sparse --second_best --doinit --dorefinement > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x3-newsampling-05-01 --num_epochs 7000 --seed 285910 --num_skills 10 --num_episodes 20 --num_steps 100 --sparse --second_best --doinit --dorefinement > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x3-newsampling-05-01 --num_epochs 7000 --seed 105399 --num_skills 10 --num_episodes 20 --num_steps 100 --sparse --second_best --doinit --dorefinement > /dev/null 2>&1 &
nohup python3 sac-fm_training.py --env_name 2x3-newsampling-05-01 --num_epochs 7000 --seed 195738 --num_skills 10 --num_episodes 20 --num_steps 100 --sparse --second_best --doinit --dorefinement > /dev/null 2>&1 &