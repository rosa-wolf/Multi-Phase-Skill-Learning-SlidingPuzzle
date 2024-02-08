#!/bin/bash

nohup python3 sac-train.py --env_name 2x2 --num_epochs 2000 --sparse --seed 12345 > log_test.out 2>&1 &
