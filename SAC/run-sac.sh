#!/bin/bash

nohup python3 sac-fm_training.py  --num_epochs 10000 --env_name fm-policy-parallel > /dev/null 2>&1 &