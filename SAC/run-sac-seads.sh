#!/bin/bash


nohup python3 SEADS-train.py --env_name seads_2x3-relabeling --num_epochs 10000 --seed 123456 --num_skills 2 --num_episodes 20 --num_steps 100 --relabeling --novelty_bonus --second_best > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x3-nonovelty --num_epochs 10000 --seed 123456 --num_skills 3 --num_episodes 20 --num_steps 100 --second_best  > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x3-nosecondbest --num_epochs 10000 --seed 123456 --num_skills 3 --num_episodes 20 --num_steps 100 --novelty_bonus  > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x3-nobonuses--num_epochs 10000 --seed 123456 --num_skills 3 --num_episodes 20 --num_steps 100 > /dev/null 2>&1 &