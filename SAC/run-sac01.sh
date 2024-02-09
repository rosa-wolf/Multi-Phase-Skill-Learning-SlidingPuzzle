#!/bin/bash

nohup python3 sac-train-old.py --env_name 2x2_prio-buffer --num_epochs 2000 --sparse --seed 12345 > log_test.out 2>&1 &
