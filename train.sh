#!/bin/bash

nohup python3 train_fm+policy.py --env_name fm-rl-parallel --updates_per_epoch 10 --num_epochs 10000 --cuda --lr 0.005 > /dev/null 2>&1 &