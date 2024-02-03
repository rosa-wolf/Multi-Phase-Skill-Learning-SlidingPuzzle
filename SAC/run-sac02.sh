#!/bin/bash

nohup  python3 sac-train.py --env_name 2x2_test-obs-coord --num_epochs 10000 --reward_on_change --sparse --seed 12345 --include_box_pos > /dev/null 2>&1 &
nohup  python3 sac-train.py --env_name 2x2_test-obs-coord --num_epochs 10000 --reward_on_change --sparse --seed 48719 --include_box_pos > /dev/null 2>&1 &
nohup  python3 sac-train.py --env_name 2x2_test-obs-nocoord --num_epochs 10000 --reward_on_change --sparse --seed 12345 --include_box_pos > /dev/null 2>&1 &
nohup  python3 sac-train.py --env_name 2x2_test-obs-nocoord --num_epochs 10000 --reward_on_change --sparse --seed 48719 --include_box_pos > /dev/null 2>&1 &