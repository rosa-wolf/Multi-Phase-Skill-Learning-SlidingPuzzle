#!/bin/bash

nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma95 --num_epochs 3000 --num_skills 1 --seed 123456 --sparse --gamma 0.95 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma93 --num_epochs 3000 --num_skills 1 --seed 123456 --sparse --gamma 0.93 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma91 --num_epochs 3000 --num_skills 1 --seed 123456 --sparse --gamma 0.91 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma89 --num_epochs 3000 --num_skills 1 --seed 123456 --sparse --gamma 0.89 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma86 --num_epochs 3000 --num_skills 1 --seed 123456 --sparse --gamma 0.86 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma90 --num_epochs 3000 --num_skills 1 --seed 123456 --sparse --gamma 0.9 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma97 --num_epochs 3000 --num_skills 1 --seed 123456 --sparse --gamma 0.97 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma99 --num_epochs 3000 --num_skills 1 --seed 123456 --sparse --gamma 0.99 --seed 123456 > /dev/null 2>&1 &