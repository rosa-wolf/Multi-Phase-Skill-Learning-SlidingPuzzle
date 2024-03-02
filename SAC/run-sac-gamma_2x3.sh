#!/bin/bash

nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma98 --num_epochs 3000 --lr 0.0005 --num_skills 1 --seed 123456 --sparse --gamma 0.98 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma80 --num_epochs 3000 --lr 0.0005 --num_skills 1 --seed 123456 --sparse --gamma 0.80 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma81 --num_epochs 3000 --lr 0.0005 --num_skills 1 --seed 123456 --sparse --gamma 0.81 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma82 --num_epochs 3000 --lr 0.0005 --num_skills 1 --seed 123456 --sparse --gamma 0.82 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma83 --num_epochs 3000 --lr 0.0005 --num_skills 1 --seed 123456 --sparse --gamma 0.83 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma84 --num_epochs 3000 --lr 0.0005 --num_skills 1 --seed 123456 --sparse --gamma 0.84 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma85 --num_epochs 3000 --lr 0.0005 --num_skills 1 --seed 123456 --sparse --gamma 0.85 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma86 --num_epochs 3000 --lr 0.0005 --num_skills 1 --seed 123456 --sparse --gamma 0.86 --seed 123456 > /dev/null 2>&1 &