#!/bin/bash

nohup python3 train_fm+policy.py --env_name fm-rl-parallel_new --updates_per_epoch 500 --num_epochs 10000 --cuda --lr 0.001 --automatic_entropy_tuning > /dev/null 2>&1 &