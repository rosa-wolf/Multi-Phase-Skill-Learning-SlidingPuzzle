"""
discrete version of single_policy_env to improve RL
- currently training of single_policy_env is not successful
- thus we discretize the actions for now
- to this end we do eef-position control and have only a finite number of positions on the board the actor can choose from
- to make it even simpler we start by setting the joint position (without any form of realistic control)
- the goal is to gradually decrease the distance between those positions to converge to the continuous setting again
"""

from typing import Optional, Tuple, Dict, Any

import gym
import numpy as np
import time
from gym.core import ObsType, ActType
from gym.spaces import Box, Dict, Discrete, MultiBinary
from gym.utils import seeding
from puzzle_scene import PuzzleScene
from robotic import ry
import torch

from forwardmodel.forward_model import ForwardModel


class PuzzleEnv(gym.Env):
    """
    Custom environment for the sliding puzzle
    """

    def __init__(self,
                 path='slidingPuzzle.g',
                 max_steps=100,
                 evaluate=False,
                 random_init_pos=True,
                 random_init_config=True,
                 random_init_board=False,
                 penalize=False,
                 give_sym_obs=False,
                 sparse_reward=False,
                 reward_on_end=False,
                 term_on_change=False,
                 verbose=0):

        """
        Args:
        :param path: path to scene file
        :param max_steps: Maximum number of steps per episode
        :param evaluate: evaluation mode (default false)
        :param random_init_pos:     whether agent should be placed in random initial position on start of episode (default true)
        :param random_init_config:  whether puzzle pieces should be in random positions initially (default true)
        :param random_init_board:   whether to NOT ensure that skill execution is possible in initial board configuration (default false)
        :param penalize:            whether to penalize going down in places where this does not lead to change in symbolic observation (default false)
        :param give_sym_obs:        whether to give symbolic observation in agents observation (default false)
        :param sparse_reward:       whether to only give a reward on change of symbolic observation (default false)
        :param reward_on_end:       whether to only give reward on episode termination (default false)
                                    - only use in combination with term_on_change=True
        :param term_on_change:      whether to terminate episode on change of symbolic observation (default false)
        :param verbose:             whether to render scene (default false)
        """
        # ground truth skills
        self.skills = np.array([[1, 0], [3, 0], [0, 1], [2, 1], [4, 1], [1, 2], [5, 2],
                                [0, 3], [4, 3], [1, 4], [3, 4], [5, 4], [2, 5], [4, 5]])

        # parameters to control initial env configuration
        self.random_init_pos = random_init_pos
        self.random_init_config = random_init_config
        self.random_init_board = random_init_board

        # parameters to control reward
        self.penalize = penalize
        self.give_sym_obs = give_sym_obs
        self.sparse_reward = sparse_reward
        self.reward_on_end = reward_on_end

        # if we only give reward on last episode then we should terminate on change of symbolic observation
        if self.reward_on_end:
            self.term_on_change = True
        else:
            self.term_on_change = term_on_change

        # evaluation mode (if true terminate scene on change of symbolic observation)
        # for evaluating policy on a visual level
        self.evaluate = evaluate

        # has actor fulfilled criteria of termination
        self.terminated = False
        self._max_episode_steps = max_steps
        self.env_step_counter = 0
        self.episode = 0

        # initialize scene
        self.scene = PuzzleScene(path, verbose=verbose)

        # initialize action_space
        # discrete action space: first parameter chooses between one of 14 optimal positions where one skill can be
        # executed, second is 0 if push movement should not be executed and 1 if it should
        self.action_space = Box(low=np.array([0, 0]), high=np.array([14, 1]), shape=(2,), dtype=np.float64)

        # store symbolic observation from previous step to check for change in symbolic observation
        # and for calculating reward based on forward model
        self._old_sym_obs = self.scene.sym_state.copy()

        # variable to remember whether actor was setback because it went outside of boundary
        # (currently not in use)
        self._setback = False

        # store which skill actor is executing, to give reward accordingly
        # (as we have a skill-conditioned policy)
        self._skill = None

        # opt push position for all 14 skills
        self.opt_pos = np.array([[0.059, -0.09],
                                 [-0.13, 0.14],
                                 [-0.195, -0.09],
                                 [0.195, -0.09],
                                 [0, 0.14],
                                 [-0.059, -0.09],
                                 [0.13, 0.14],
                                 [-0.13, -0.14],
                                 [0.057, 0.09],
                                 [0., -0.14],
                                 [-0.195, 0.09],
                                 [0.195, 0.09],
                                 [0.13, -0.14],
                                 [-0.057, 0.09]])
        self.opt_pos = np.concatenate((self.opt_pos,
                                       np.repeat(np.array([[self.scene.q0[2], self.scene.q0[3]]]),
                                                 self.opt_pos.shape[0], axis=0)),
                                      axis=1)

        # sanity check
        print(self.opt_pos.shape)



        # load fully trained forward model
        self.fm = ForwardModel(batch_size=60, learning_rate=0.001)
        self.fm.model.load_state_dict(torch.load("../SEADS_SlidingPuzzle/forwardmodel/models/best_model"))
        self.fm.model.eval()

    @property
    def skill(self) -> int:
        return self._skill

    @skill.setter
    def skill(self, value: int):
        self._skill = value

    def step(self, action: Dict) -> tuple[Dict, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        :param action: an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # store current symbolic observation before executing action
        self._old_sym_obs = self.scene.sym_state.copy()

        # apply action
        self.apply_action(action)

        # check if joints are outside observation_space
        # if so set back to last valid position
        self._setback = not self.scene.check_limits()

        obs = self._get_observation()

        if not self.reward_on_end:
            # get reward (if we dont only want to give reward on last step of episode)
            reward = self._reward(action)

        # check if symbolic observation changed
        if not (self._old_sym_obs == self.scene.sym_state).all():
            # for episode termination on change of symbolic observation
            if self.term_on_change:
                self.terminated = True

                if self.reward_on_end:
                    reward = self._reward(action)
            else:
                # setback to previous state to continue training until step limit is reached
                # make sure reward and obs is calculated before this state change
                self.scene.sym_state = self._old_sym_obs
                self.scene.set_to_symbolic_state()

        # look whether conditions for termination are met
        # make sure to reset env in trainings loop if done
        done = self._termination()
        if not done:
            self.env_step_counter += 1
            # set puzzle pieces back to discrete places (in case they were moved just a little bit
            # but not enough do change symbolic observation)
            # self.scene.set_to_symbolic_state()

        # caution: dont return resetted values but values that let to reset
        return obs, reward, done, {}

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
        self.episode += 1

        if self.random_init_pos:
            # Set agent to random initial position inside a box
            init_pos = np.random.uniform(-0.3, .3, (2,))
            self.scene.q = [init_pos[0], init_pos[1], self.scene.q0[2], self.scene.q0[3]]
        if self.random_init_config:
            # should it be possible to apply skill on initial board configuration?
            if self.random_init_board:
                # randomly pick the field where no block is initially
                field = np.delete(np.arange(0, 6), np.random.choice(np.arange(0, 6)))
            else:
                # take initial empty field such that skill execution is possible
                field = np.delete(np.arange(0, 6), self.skills[self.skill, 1])

            # put blocks in random fields, except the one that has to be free
            order = np.random.permutation(field)
            sym_obs = np.zeros((5, 6))
            for i in range(5):
                sym_obs[i, order[i]] = 1

            self.scene.sym_state = sym_obs
            self.scene.set_to_symbolic_state()

        self._old_sym_obs = self.scene.sym_state.copy()

        return self._get_observation()

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
            return True
        return False

    def _get_observation(self):
        """
        Returns the observation: Robot joint states and symbolic observation
        """
        q, q_dot, sym_obs = self.scene.state
        if self.give_sym_obs:
            # symbolic observation is part of agents observation
            return np.concatenate(((q[:2]), sym_obs.flatten()))

        return q[:2]


    @property
    def observation_space(self):
        """
        Defines bounds of the observation space (Hard coded for now)
        """
        # Todo: implement reading out actual limits from file
        shape = 2

        # if symbolic observation is part of agents observation
        # 0-2 joint position in x,y,z
        # 3, 4: velocity in x,y direction
        # 5 - end: symbolic observation (flattened)
        if self.give_sym_obs:
            shape = 2 + self.scene.sym_state.shape[0] * self.scene.sym_state.shape[1]

        return Box(low=-np.inf, high=np.inf, shape=(shape,), dtype=np.float64)

    def apply_action(self, action):
        """
        Applys the action in the scene
        :param action: 2D velocity, and [-1, 1] indicating whether we want to  execute skill afterward
        :return:
        """
        # get position actor chose to go to
        # set this joint pos hard
        pos = int(action[0])
        if pos == 14:
            pos = 13

        self.scene.q = self.opt_pos[pos]

        if action[1] >= 0.5:
            # execute skill independent of where current position of actor
            self.execute_skill()


    def execute_skill(self):
        """
        Hard coded skill to move one block forward onto free space
        """
        prev_joint_pos = self.scene.q
        prev_joint_pos[2] = self.scene.q0[2]

        prev_board_state = self.scene.sym_state.copy()

        # skills where orientation of end-effector does not has to be changed for
        no_orient_change = [1, 4, 6, 7, 9, 12]

        if self.skill not in no_orient_change:
            print("orientation change")
            # change orientation by pi/2
            self.scene.v = np.array([0., 0., 0.1, 1.7])
            self.scene.velocity_control(1000)

        # move to right z-position (move down)
        self.scene.v = np.array([0., 0., -0.15, 0.])
        self.scene.velocity_control(650)

        if self.skill == 7 or self.skill == 9 or self.skill == 12:
            # go forward
            self.scene.v = np.array([0., 0.15, 0., 0.])
        elif self.skill == 1 or self.skill == 4 or self.skill == 6:
            # go backward
            self.scene.v = np.array([0., -0.15, 0., 0.])
        elif self.skill == 2 or self.skill == 5 or self.skill == 10 or self.skill == 13:
            # go to left
            self.scene.v = np.array([0.15, 0., 0., 0.])
        else:
            # go to right
            self.scene.v = np.array([-0.15, 0., 0., 0.])

        change = self.scene.velocity_control(1300)

        if self.evaluate:
            # if we just want to look at the model, we may want to stop movement if it was successfull
            self.terminated = change

        # set robot back to position before skill execution (such that reward will be calculated correctly)
        self.scene.C.setJointState(prev_joint_pos)
        self.scene.S.setState(self.scene.C.getFrameState())
        self.scene.S.step([], self.scene.tau, ry.ControlMode.none)

    def _reward(self, action) -> float:
        """
        Calculates reward, which is based on symbolic observation change
        """
        # for sparse reward
        if self.sparse_reward:
            # only give reward on change of symbolic observation
            reward = 0
        else:
            # reward shaping: give reward that is linearly dependent
            # on distance to some optimal position where agent should ideally execute push movement
            # -> reward positive, linear and higher the smaller the distance to optimal point is
            # use when fixed episode number because otherwise agent
            # can accumulate higher reward, by doing non-terminating actions

            # define place of highest reward for each skill
            if self.skill == 0:
                opt = np.array([0.059, -0.09, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (lower right corner)
                max = np.array([-0.2, 0.2, self.scene.q0[2], self.scene.q0[3]])
            elif self.skill == 1:
                opt = np.array([-0.13, 0.14, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (upper left corner)
                max = np.array([0.2, -0.2, self.scene.q0[2], self.scene.q0[3]])
            elif self.skill == 2:
                opt = np.array([-0.195, -0.09, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (lower left corner)
                max = np.array([0.2, 0.2, self.scene.q0[2], self.scene.q0[3]])
            elif self.skill == 3:
                opt = np.array([0.195, -0.09, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (lower right corner)
                max = np.array([-0.2, 0.2, self.scene.q0[2], self.scene.q0[3]])
            elif self.skill == 4:
                opt = np.array([0., 0.14, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (upper left corner)
                max = np.array([0.2, -0.2, self.scene.q0[2], self.scene.q0[3]])
            elif self.skill == 5:
                opt = np.array([-0.059, -0.09, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (lower left corner)
                max = np.array([0.2, 0.2, self.scene.q0[2], self.scene.q0[3]])
            elif self.skill == 6:
                opt = np.array([0.13, 0.14, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (upper right corner)
                max = np.array([-0.2, -0.2, self.scene.q0[2], self.scene.q0[3]])
            elif self.skill == 7:
                opt = np.array([-0.13, -0.14, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (lower left corner)
                max = np.array([0.2, 0.2, self.scene.q0[2], self.scene.q0[3]])
            elif self.skill == 8:
                opt = np.array([0.057, 0.09, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (upper right corner)
                max = np.array([-0.2, -0.2, self.scene.q0[2], self.scene.q0[3]])
            elif self.skill == 9:
                opt = np.array([0., -0.14, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (lower left corner)
                max = np.array([0.2, 0.2, self.scene.q0[2], self.scene.q0[3]])
            elif self.skill == 10:
                opt = np.array([-0.195, 0.09, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (upper left corner)
                max = np.array([0.2, -0.2, self.scene.q0[2], self.scene.q0[3]])
            elif self.skill == 11:
                opt = np.array([0.195, 0.09, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (upper right corner)
                max = np.array([-0.2, -0.2, self.scene.q0[2], self.scene.q0[3]])
            elif self.skill == 12:
                opt = np.array([0.13, -0.14, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (lower right corner)
                max = np.array([-0.2, 0.2, self.scene.q0[2], self.scene.q0[3]])
            elif self.skill == 13:
                opt = np.array([-0.057, 0.09, self.scene.q0[2], self.scene.q0[3]])
                # loc with max distance to optimum (upper left corner)
                max = np.array([0.2, -0.2, self.scene.q0[2], self.scene.q0[3]])

            #max = np.array([-0.2, 0.2, self.scene.q0[2], self.scene.q0[3]])#  location with the lowest reward (lower right corner)
            loc = self.scene.C.getJointState()  # current location

            # reward: max distance - current distance
            reward = np.linalg.norm(opt - max) - np.linalg.norm(opt - loc)

            # negative reward that considers angle and distance (can also be used without fixed episode number
            # (because it gives incentive to make episodes as short as possible)
            #opt = np.array([0.11114188, -0.09727765, self.scene.q0[2], self.scene.q0[3]])  # loc with the highest reward (reward = 0)
            #loc = self.scene.C.getJointState()  # current location
            ## distance to opt
            #reward = - np.linalg.norm(opt - loc)
            ## if y smaller than that of location take sin (else take times 1)
            #if loc[1] < opt[1]:
            #    h = np.array([opt[0], loc[1], self.scene.q0[2], self.scene.q0[3]])  # helper point to calculate sin of angle
            #    reward *= np.linalg.norm(loc - h) / np.linalg.norm(loc - opt)

        # give reward on every change of symbolic observation according to forward model
        if not (self._old_sym_obs == self.scene.sym_state).all():
            reward += self.fm.calculate_reward(self._old_sym_obs.flatten(),
                                               self.scene.sym_state.flatten(),
                                               self.skill)
        elif self.penalize:
            # if we penalize also give reward when no change of symbolic obervation occured
            # but agent tried to execute push movement
            if action[1] >= 0.5:
                reward += 0.0001 * self.fm.calculate_reward(self._old_sym_obs.flatten(),
                                                            self.scene.sym_state.flatten(),
                                                            self.skill)
        return reward

