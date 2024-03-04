#!/bin/bash

nohup python3 SEADS-train.py --env_name seads_1x2_gradsteps16 --gradient_steps 16 --num_epochs 5000 --seed 123456 --num_skills 2 --num_episodes 20 --num_steps 100 --relabeling > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x2_gradsteps16 --gradient_steps 16 --num_epochs 5000 --seed 123456 --num_skills 2 --num_episodes 20 --num_steps 100 --relabeling > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x3_gradsteps16 --gradient_steps 16 --num_epochs 6000 --seed 123456 --num_skills 3 --num_episodes 20 --num_steps 100 --relabeling > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_1x2_gradsteps16 --gradient_steps 16 --num_epochs 6000 --seed 123456 --num_skills 2 --num_episodes 20 --num_steps 100  > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x2_gradsteps16 --gradient_steps 16 --num_epochs 10000 --seed 123456 --num_skills 2 --num_episodes 20 --num_steps 100  > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x3_gradsteps16 --gradient_steps 16 --num_epochs 10000 --seed 123456 --num_skills 3 --num_episodes 20 --num_steps 100  > /dev/null 2>&1 &