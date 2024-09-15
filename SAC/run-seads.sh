#!/bin/bash

nohup  python3 SEADS-train.py --env_name 2x3 --num_epochs 8000 --num_skills 3 --novelty_bonus --second_best > /dev/null 2>&1 &
nohup  python3 SEADS-train.py --env_name 2x3 --num_epochs 8000 --num_skills 5 --novelty_bonus --second_best > /dev/null 2>&1 &
nohup  python3 SEADS-train.py --env_name 2x3 --num_epochs 8000 --num_skills 10 --novelty_bonus --second_best  > /dev/null 2>&1 &
nohup  python3 SEADS-train.py --env_name 2x3 --num_epochs 8000 --num_skills 15 --novelty_bonus --second_best  > /dev/null 2>&1 &
nohup  python3 SEADS-train.py --env_name 2x3 --num_epochs 8000 --num_skills 20 --novelty_bonus --second_best  > /dev/null 2>&1 &