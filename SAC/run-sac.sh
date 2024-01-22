#!/bin/bash

nohup python3 sac-train.py --env_name skill_conditioned_2x2_eval01 --term_on_change --reward_on_change --sparse --seed 1234567 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_2x2_eval02 --term_on_change --reward_on_change --sparse --seed 5847836 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_2x2_eval03 --term_on_change --reward_on_change --sparse --seed 8124099 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_2x2_eval04 --term_on_change --reward_on_change --sparse --seed 2351294 > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_2x2_eval05 --term_on_change --reward_on_change --sparse --seed 9148235 > /dev/null 2>&1 &