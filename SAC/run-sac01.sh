#!/bin/bash

nohup python3 sac-train-old.py --env_name skill_conditioned_2x2_prio-buffer --num_epochs 2000 --sparse --seed 12345 > /dev/null 2>&1 &
