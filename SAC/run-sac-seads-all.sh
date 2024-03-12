#!/bin/bash


nohup python3 SEADS-train.py --env_name seads_2x3-only-novelty --num_epochs 4000 --seed 123456 --num_skills 3 --num_episodes 20 --num_steps 200 --novelty_bonus > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x3-only-novelty --num_epochs 4000 --seed 578907 --num_skills 3 --num_episodes 20 --num_steps 200 --novelty_bonus > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x3-only-novelty --num_epochs 4000 --seed 285910 --num_skills 3 --num_episodes 20 --num_steps 200 --novelty_bonus > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x3-only-novelty --num_epochs 4000 --seed 105399 --num_skills 3 --num_episodes 20 --num_steps 200 --novelty_bonus > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x3-only-novelty --num_epochs 4000 --seed 195738 --num_skills 3 --num_episodes 20 --num_steps 200 --novelty_bonus > /dev/null 2>&1 &
