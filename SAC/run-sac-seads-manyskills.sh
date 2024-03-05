#!/bin/bash


nohup python3 SEADS-train.py --env_name seads_2x2-2skills --num_epochs 5000 --seed 123456 --num_skills 2 --num_episodes 20 --num_steps 100 --novelty_bonus --second_best > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x2-3skills --num_epochs 5000 --seed 123456 --num_skills 3 --num_episodes 20 --num_steps 100 --novelty_bonus --second_best > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x2-4skills --num_epochs 5000 --seed 123456 --num_skills 4 --num_episodes 20 --num_steps 100 --novelty_bonus --second_best > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x2-5skills --num_epochs 5000 --seed 123456 --num_skills 5 --num_episodes 20 --num_steps 100 --novelty_bonus --second_best > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x2-6skills --num_epochs 5000 --seed 123456 --num_skills 6 --num_episodes 20 --num_steps 100 --novelty_bonus --second_best > /dev/null 2>&1 &
nohup python3 SEADS-train.py --env_name seads_2x2-7skills --num_epochs 5000 --seed 123456 --num_skills 7 --num_episodes 20 --num_steps 100 --novelty_bonus --second_best > /dev/null 2>&1 &