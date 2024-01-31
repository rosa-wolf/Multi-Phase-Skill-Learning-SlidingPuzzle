#!/bin/bash

nohup  python3 sac-fm_training_lookup.py --env_name parallel_2x2_lookup-table --sparse --num_skills 2 > /dev/null 2>&1 &
