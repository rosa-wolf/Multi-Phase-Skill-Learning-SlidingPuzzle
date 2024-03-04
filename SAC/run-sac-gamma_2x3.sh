#!/bin/bash

nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma95lr0005-smallerbox --num_epochs 3000 --lr 0.0005 --num_skills 1 --seed 123456 --sparse --gamma 0.95 --seed 123456 > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3-1skill-gamma95lr001-smallerbox --num_epochs 3000 --lr 0.001 --num_skills 1 --seed 123456 --sparse --gamma 0.95 --seed 123456 > /dev/null 2>&1 &