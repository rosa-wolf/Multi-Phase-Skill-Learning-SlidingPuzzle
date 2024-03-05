#!/bin/bash

nohup python3 SEADS-train.py --env_name 2x3-relabeling --num_epochs 10000 --seed 123456 --num_skills 3chm --num_episodes 20 --num_steps 100 --sparse --relabeling > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name 2x3-no-relabeling --num_epochs 10000 --seed 123456 --num_skills 3 --num_episodes 20 --num_steps 100 --sparse > /dev/null 2>&1 &