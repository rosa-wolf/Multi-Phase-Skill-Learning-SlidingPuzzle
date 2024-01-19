#!/bin/bash

nohup  python3 sac-fm_training.py --env_name parallel_2x2 --num_skills 2 --reward_on_end --term_on_change > /dev/null 2>&1 &