#!/bin/bash


nohup python3 sac-train-new.py --env_name 2x3_3skills_gamma099 --num_epochs 2000 --dict_obs --seed 123456 --sparse --gamma 0.99 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3_3skills_gamma099 --num_epochs 2000 --dict_obs --seed 487193 --sparse --gamma 0.99 > /dev/null 2>&1 &