#!/bin/bash


nohup python3 sac-train-new.py --env_name 2x3_3skills --num_epochs 3000 --num_skills 3 --seed 123456 --sparse > /dev/null 2>&1 &
nohup python3 sac-train-new.py --env_name 2x3_3skills --num_epochs 3000 --num_skills 3 --seed 487193 --sparse > /dev/null 2>&1 &