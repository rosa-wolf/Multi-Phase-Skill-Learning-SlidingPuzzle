#!/bin/bash

nohup python3 sac-train.py --env_name skill_conditioned_2x2_eval01 --term_on_change --reward_on_change --sparse > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_2x2_eval02 --term_on_change --reward_on_change --sparse > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_2x2_eval03 --term_on_change --reward_on_change --sparse > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_2x2_eval04 --term_on_change --reward_on_change --sparse > /dev/null 2>&1 &
nohup python3 sac-train.py --env_name skill_conditioned_2x2_eval05 --term_on_change --reward_on_change --sparse > /dev/null 2>&1 &