#!/bin/bash

nohup python3 train_fm+policy.py --env_name fm-rl-parallel_add-reward_auto-tuning --updates_per_epoch 500 --num_epochs 10000 --num_start_epis 800 --cuda --lr 0.001 --automatic_entropy_tuning --target_entropy -2.5 > /dev/null 2>&1 &
nohup python3 train_fm+policy.py --env_name fm-rl-parallel_add-reward_alpha-05 --updates_per_epoch 500 --num_epochs 10000 --num_start_epis 800 --cuda --lr 0.001 --alpha 0.4  > /dev/null 2>&1 &