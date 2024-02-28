#!/bin/bash


nohup python3 sac-fm_training.py --env_name 3x3-prior --num_skills 4 --prior_buffer --num_episodes 20 > /dev/null 2>&1 &