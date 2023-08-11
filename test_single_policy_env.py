"""
Test single_polciy_env

- test reward function with different versions of reward
- test action
- test joint state setting
- test resetting
- test symbolic observation and previus symbolic observation
- test different variants of observation space
"""

import numpy as np
import time
from robotic import ry
from gym.utils.env_checker import check_env
from gymframework.single_policy_env import PuzzleEnv



SKILLS = np.array([[1, 0], [3, 0], [0, 1],
                   [2, 1], [4, 1], [1, 2],
                   [5, 2], [0, 3], [4, 3],
                   [1, 4], [3, 4], [5, 4],
                   [2, 5], [4, 5]])

if __name__ == '__main__':
    # TODO: implement eef-position control
    # sanity check of custom env
    env = PuzzleEnv(verbose=1)
    env.skill = 13
    env.reset()
    print(env.scene.sym_state)
    # go to correct x-y-position
    env.scene.q = np.array([-0.057, 0.09, env.scene.q0[2], env.scene.q0[3]])
    time.sleep(3)
    env.execute_skill()