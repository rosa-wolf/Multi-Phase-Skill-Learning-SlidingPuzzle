#!/bin/bash

nohup  python3 sac-fm_training.py --env_name 2x3 --num_epochs 8000 --num_skills 3 --sparse --second_best --doinit --dorefinement > /dev/null 2>&1 &
nohup  python3 sac-fm_training.py --env_name 2x3 --num_epochs 8000 --num_skills 5 --sparse --second_best --doinit --dorefinement > /dev/null 2>&1 &
nohup  python3 sac-fm_training.py --env_name 2x3 --num_epochs 8000 --num_skills 10 --sparse --second_best --doinit --dorefinement > /dev/null 2>&1 &
nohup  python3 sac-fm_training.py --env_name 2x3 --num_epochs 8000 --num_skills 15 --sparse --second_best --doinit --dorefinement > /dev/null 2>&1 &
nohup  python3 sac-fm_training.py --env_name 2x3 --num_epochs 8000 --num_skills 20 --sparse --second_best --doinit --dorefinement > /dev/null 2>&1 &