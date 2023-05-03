from typing import Optional, Tuple, Dict, Any

import gym
import numpy as np
import time
from gym.core import ObsType, ActType
from gym.spaces import Box, Dict, Discrete, MultiBinary
from gym.utils import seeding
from puzzle_scene import PuzzleScene



class PuzzleEnv(gym.Env):
    """
    Custom environment for the sliding puzzle
    """

    def __init__(self,
                 path='slidingPuzzle.g',
                 max_steps=2000,
                 random_init_pos=False,
                 nsubsteps=1):

        """
        Args:
        :param max_steps: Maximum number of steps per episode
        :param random_init_pos: whether agent should be placed in random initial position on start of episode
        """

        # has actor fullfilled criteria of termination
        self.terminated = False
        self.env_step_counter = 0
        self._max_episode_steps = max_steps
        self.episode = 0

        self.scene = PuzzleScene(path)

        self.dt = self.scene.tau

        self.random_init_pos = random_init_pos

        # range in which robot should be able to move
        # velocities in x-y direction (no change in orientation for now)
        # Todo: make action space 1d array for sac algorithm
        #self.action_space = Dict({"q_dot": Box(low=-1., high=1., shape=(2,), dtype=np.float64),
        #                          "perform_skill": Discrete(2)})

        # first two is velocity in x-y plane, last decides wether we perform skill (>0) or not (<=0)
        self.action_space = Box(low=-1., high=1., shape=(3,), dtype=np.float64)

        # store previous observation and observation
        self._old_obs = self._get_observation()
        self._obs = self._get_observation()

        self.reset()

    def step(self, action: Dict) -> tuple[Dict, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # apply action
        #prev_q = self._obs["q"]
        prev_q = self._obs[:3]
        self.apply_action(action)
        # check if joints are outside observation_space
        if not self.scene.check_limits():
            # go back to previous position (orientation cannot change currently)
            self.scene.q = np.concatenate((prev_q, np.array([self.scene.q0[3]])))

        # if so set back to last valid position

        # get observation
        self._old_obs = self._obs
        self._obs = self._get_observation()

        # get reward
        reward = self._reward()
        #print("reward = ", reward)

        # look whether conditions for termination are met
        done = self._termination()
        if not done:
            self.env_step_counter += 1

        return self._obs, reward, done, {}

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None,) -> tuple[dict[str, Any], dict[Any, Any]]:
        """
        Resets the environment (including the agent) to the initial conditions.
        """

        super().reset(seed=seed)
        self.scene.reset()
        self.terminated = False
        self.env_step_counter = 0

        if self.random_init_pos:
            # Todo: Set agent to random position

            pass

        self._old_obs = self._get_observation()
        self._obs = self._get_observation()

        return self._obs, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _termination(self):
        """
        Checks if the robot either exceeded the maximum number of steps or is terminated according to another task
        dependent metric.
        Returns:
            True if robot should terminate
        """
        if self.terminated or self.env_step_counter >= self._max_episode_steps:
            self.terminated = False
            self.env_step_counter = 0
            self.episode += 1
            return True
        return False

    def _get_observation(self):
        """
        Returns the observation: Robot joint states and velocites and symbolic observation
        """
        q, q_dot, sym_obs = self.scene.state
        return np.concatenate((q[:3], q_dot[:2], sym_obs.flatten()))
        #return {"q": np.array(q)[:3], "q_dot": np.array(q_dot)[:2], "sym_obs": sym_obs}

    @property
    def observation_space(self):
        """
        Defines bounds of the observation space (Hard coded for now)
        """
        # Todo: implement reading out actual limits from file
        # get joint limits (only of translational joints for now)
        low_q = -5.
        high_q = 5.

        shape = 5 + self.scene.sym_state.shape[0] * self.scene.sym_state.shape[1]
        # Todo: make observation spae one single array (such that it works with sac algorithm)
        # 0-2 joint position in x,y,z
        # 3, 4: velocity in x,y direction
        # 5 - end: symbolic observation (flattened)
        return Box(low=-np.inf, high=np.inf, shape=(shape,), dtype=np.float64)

        #return Dict({"q": Box(low=low_q, high=high_q, shape=(3,), dtype=np.float64),
        #             "q_dot": Box(low=0., high=1., shape=(2, ), dtype=np.float64),
        #             "sym_obs": MultiBinary(n=self.scene.sym_state.shape)})

    def apply_action(self, action):
        """
        Applys the action in the scene
        :param action: 2D velocity, and [-1, 1] indicating whether we want to  execute skill afterwards
        :return:
        """
        # Todo: Maybe it makes more sense to include #steps in action for which to apply the velocities
        # read out action (velocities in x-y plane)
        #self.scene.v = np.array([action["q_dot"][0], action["q_dot"][1], 0., 0.])
        self.scene.v = np.array([action[0], action[1], 0., 0.])
        # apply action (for now always for (just) ten steps)
        self.scene.velocity_control(1)
        # execute skill if action says so
        #if action["perform_skill"] == 1:
        #    self.execute_skill()

        if action[2] > 0:
            self.execute_skill()

    def _reward(self) -> float:
        """
        Calculates reward, which is based on symbolic observation change
        """
        # Reward if symbolic observation changed
        # if symbolic observation is invalid return negative reward
        # if symbolic observation changed and is valid return positive reward
        # which is dependent on steps needed to reach this state

        if not self.scene.valid_state():
            self.terminated = True
            return -1

        #if (self._old_obs["sym_obs"] == self._obs["sym_obs"]).all():
        if not (self._old_obs[5:] == self._obs[5:]).all():
            self.terminated = True
            if self.env_step_counter != 0:
                return 10./self.env_step_counter
            else:
                return 10.

        return 0

    def execute_skill(self):
        """
        Hard coded skill to move one block forward onto free space
        """
        # Todo:
        # check if field we are trying to push onto is free
        # look at current x-y position of end-effector
        # estimate which block is going to be pushed
        # look whether neighboring place is occupied

        # move to right z-position (move down)
        self.scene.v = np.array([0., 0., -0.05, 0.])
        self.scene.velocity_control(30)

        # go forward
        self.scene.v = np.array([0., 0.04, 0., 0.])
        self.scene.velocity_control(50)
