#!/bin/bash

nohup python3 sac-train.py --env_name skill_conditioned_1x2_eval01 --num_skills 2 --num_epochs 1000 --seed 1234567 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_1x2_eval02 --num_skills 2 --num_epochs 1000 --seed 5847836 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_1x2_eval03 --num_skills 2 --num_epochs 1000 --seed 8124099 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_1x2_eval04 --num_skills 2 --num_epochs 1000 --seed 4657382 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_1x2_eval05 --num_skills 2 --num_epochs 1000 --seed 9038572 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_1x2_eval06 --num_skills 2 --num_epochs 1000 --seed 1250681 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_1x2_eval07 --num_skills 2 --num_epochs 1000 --seed 2140829 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_1x2_eval08 --num_skills 2 --num_epochs 1000 --seed 3287104 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_1x2_eval09 --num_skills 2 --num_epochs 1000 --seed 3949683 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_1x2_eval10 --num_skills 2 --num_epochs 1000 --seed 6832779 > /dev/null 2>&1 &
