#!/bin/bash

nohup  python3 sac-train-old.py --env_name skill_conditioned_2x2_dict --dict_obs --num_epochs 2000 --sparse --seed 123456 > /dev/null 2>&1 &
nohup  python3 sac-train-old.py --env_name skill_conditioned_2x2_dict --dict_obs --num_epochs 2000 --sparse --seed 487193 > /dev/null 2>&1 &
nohup  python3 sac-train-old.py --env_name skill_conditioned_2x2_flatBox  --num_epochs 2000 --sparse --seed 123456 > /dev/null 2>&1 &
nohup  python3 sac-train-old.py --env_name skill_conditioned_2x2_flatBox  --num_epochs 2000 --sparse --seed 487193 > /dev/null 2>&1 &