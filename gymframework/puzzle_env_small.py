from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
import time
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Box
from gymnasium.utils import seeding
from puzzle_scene import PuzzleScene
from robotic import ry
import torch

from forwardmodel_simple_input.forward_model import ForwardModel


class PuzzleEnv(gym.Env):
    """
    Custom environment for the sliding puzzle

    This is an environment to train the policy for a single skill.
    When initiating the variable for the skill is set and not changed after.
    This is only so we can use the same environment-type for all skills.

    This is the version to test out having a smaller (1x2) board
    """

    def __init__(self,
                 path='slidingPuzzle_small.g',
                 snapRatio=4.,
                 skill=0,
                 max_steps=100,
                 evaluate=False,
                 random_init_board=False,
                 puzzlesize = [1, 2],
                 sparse_reward=False,
                 reward_on_change=False,
                 reward_on_end=False,
                 term_on_change=False,
                 verbose=0):

        """
        Args:
        :param skill: which skill we want to train (influences reward and field configuration)
        :param path: path to scene file
        :param max_steps: Maximum number of steps per episode
        :param evaluate: evaluation mode (default false)
        :param random_init_board:   whether to NOT ensure that skill execution is possible in initial board configuration (default false)
        :param sparse_reward:       whether to only give a reward on change of symbolic observation (default false)
        :param reward_on_change:    whether to give additional reward when box is successfully pushed (default false)
        :param term_on_change:      whether to terminate episode on change of symbolic observation (default false)
        :param verbose:      _       whether to render scene (default false)
        """
        # ground truth skills
        # we have only one box, so there is only one skill
        self.skills = np.array([[1, 0], [0, 1]])

        # which policy are we currently training? (Influences reward)
        self.skill = skill

        # opt push position for all 14 skills (for calculating reward)
        # TODO: Make optimal position dependent on current position of block we want to push
        # position when reading out block position is the center of the block
        # depending on skill we have to push from different side on the block
        # optimal position is position at offset into right direction from the center og the block
        # offset half the side length of the block
        self.offset = 0.06
        # direction of offset [x-direction, y-direction]
        # if 0 no offset in that direction
        # -1/1: negative/positive offset in that direction
        self.opt_pos_dir = np.array([[1, 0]])
        # store which box we will push with current skill
        self.box = None

        # opt push position for all 14 skills (for calculating reward)
        #self.opt_pos = np.array([[0.06, -0.09],
        #                         [-0.05, -0.09]])

        # TODO: we cannot hardcode a position where the actor will have the maximal distance to the optimal position
        # as the optimal position changes with the position of the box
        # However, we can hardcode a maximal distance using the
        self.max = np.array([-0.25, 0.25, 0.25])
        self.max_dist = None

        # parameters to control initial env configuration
        self.random_init_board = random_init_board

        # parameters to control different versions of observation and reward
        self.sparse_reward = sparse_reward
        self.reward_on_change = reward_on_change
        self.reward_on_end = reward_on_end

        # if we only give reward on last episode then we should terminate on change of symbolic observation
        if self.reward_on_end:
            self.term_on_change = True
        else:
            self.term_on_change = term_on_change

        # has actor fulfilled criteria of termination
        self.terminated = False
        self.truncated = False
        self.env_step_counter = 0
        self._max_episode_steps = max_steps
        self.episode = 0

        # initialize scene
        self.scene = PuzzleScene(path, puzzlesize=puzzlesize, verbose=verbose, snapRatio=snapRatio)

        # desired x-y-z coordinates of eef
        self.action_space = Box(low=np.array([-1., -1., -1.]), high=np.array([1., 1., 1.]), shape=(3,))

        # store symbolic observation from previous step to check for change in symbolic observation
        # and for calculating reward based on forward model
        self._old_sym_obs = self.scene.sym_state.copy()

        self.init_sym_state = None

        # set init and goal position of box for calculating reward
        self.box_init = None
        self.box_goal = None

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

        # store current symbolic observation before executing action
        self._old_sym_obs = self.scene.sym_state.copy()

        # apply action
        self.apply_action(action)

        # check if joints are outside observation_space
        # if so set back to last valid position
        self.scene.check_limits()

        obs = self._get_observation()

        reward = self._reward()

        # check if symbolic observation changed
        if self.term_on_change:
            self.terminated = not (self._old_sym_obs == self.scene.sym_state).all()


        # look whether conditions for termination are met
        # make sure to reset env in trainings loop if done
        done = self._termination()
        if not done:
            self.env_step_counter += 1

        print("reward = ", reward)

        # caution: dont return resetted values but values that let to reset
        return obs, reward, self.terminated, self.truncated, {}

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None, ) -> tuple[dict[str, Any], dict[Any, Any]]:
        """
        Resets the environment (including the agent) to the initial conditions.
        """

        super().reset(seed=seed)
        self.scene.reset()
        self.terminated = False
        self.truncated = False
        self.env_step_counter = 0
        self.episode += 1

        # Set agent to random initial position inside a box
        init_pos = np.random.uniform(self.scene.q_lim[0, 0], self.scene.q_lim[0, 1], (2,))
        self.scene.q = [init_pos[0], init_pos[1], self.scene.q0[2], np.pi/2.]
        # should it be possible to apply skill on initial board configuration?
        if self.random_init_board:
            # randomly pick the field where no block is initially
            field = np.delete(np.arange(0, self.scene.pieces + 1),
                              np.random.choice(np.arange(0, self.scene.pieces + 1)))
        else:
            # take initial empty field such that skill execution is possible
            field = np.delete(np.arange(0, self.scene.pieces + 1), self.skills[self.skill, 1])

        # put blocks in random fields, except the one that has to be free
        order = np.random.permutation(field)
        sym_obs = np.zeros((self.scene.pieces, self.scene.pieces + 1))
        for i in range(self.scene.pieces):
            sym_obs[i, order[i]] = 1
            self.scene.sym_state = sym_obs
            self.scene.set_to_symbolic_state()

        # look which box is in the field we want to push from
        # important for reward shaping
        field = self.skills[self.skill][0]
        # get box that is currently on that field
        self.box = np.where(self.scene.sym_state[:, field] == 1)[0][0]

        self._old_sym_obs = self.scene.sym_state.copy()

        self.max_dist = np.linalg.norm((self.scene.C.getFrame("box" + str(self.box)).getPosition()).copy() - self.max)

        # set init and goal position of box
        self.box_init = (self.scene.C.getFrame("box" + str(self.box)).getPosition()).copy()
        self.box_goal = self.scene.discrete_pos[self.skills[self.skill, 1]]

        self.init_sym_state = self.scene.sym_state.copy()

        return self._get_observation(), {}

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def _termination(self):
        """
        Checks if the robot either exceeded the maximum number of steps or is terminated according to another task
        dependent metric.
        Returns:
            True if robot should terminate
        """
        if self.terminated:
            return True
        if self.env_step_counter >= self._max_episode_steps - 1:
            self.truncated = True
            return True
        return False

    def _get_observation(self):
        """
        Returns the observation:    Robot joint states and velocites and symbolic observation
                                    Executed Skill is also encoded in observation/state
        """
        q, _, _ = self.scene.state

        # normalize state space (has influence on policy, because of the added noise)
        # joint angles and box position are all in range [-0.25, 0.25]
        obs = q[:3] * 4

        # add x-and y-position of relevant puzzle piece
        pos = ((self.scene.C.getFrame("box" + str(self.box)).getPosition()).copy())[:2]

        return np.concatenate((obs, pos * 4))

    @property
    def observation_space(self):
        """
        Defines bounds of the observation space (Hard coded for now)
        """
        # Todo: implement reading out actual limits from file
        # observation space as 1D array instead
        # joint configuration (3) + skill (1)
        shape = 3  # 5 #+ self.scene.sym_state.shape[0] * self.scene.sym_state.shape[1]

        # add dimensions for position of relevant puzzle piece (x, y, z -position)
        shape += 2

        return Box(low=-1, high=1, shape=(shape,), dtype=np.float64)

    def apply_action(self, action):
        """
        Applys the action in the scene
        :param action: desired x-y-z position
        """
        # do velocity control for 100 steps
        # set velocity to go in that direction
        # (vel[0] - velocity in x-direction, vel[1] - velocity in y-direction)
        # vel[2] - velocity in z-direction (set to zero)
        # vel[3] - orientation, set to zero
        for i in range(100):
            # get current position
            act = self.scene.q[:3]

            diff = action / 4 - act

            self.scene.v = np.array([diff[0], diff[1], diff[2], 0.])
            self.scene.velocity_control(1)

    def _reward(self) -> float:
        """
        Calculates reward, which is based on symbolic observation change

        - reward shaping: give reward that is linearly dependent
          on distance to some optimal position where agent should ideally execute push movement
        -> reward positive, linear and higher the smaller the distance to optimal point is
        - use when fixed episode number because otherwise agent
          can accumulate higher reward, by doing non-terminating actions
        - define place of highest reward for each skill (optimal position)
          as outer edge of position of block we want to push
            - opt position changes when block position changes

        """

        # TODO: change max to correct value
        reward = 0
        if not self.sparse_reward:
            # read out position of box that should be pushed
            box_pos = (self.scene.C.getFrame("box" + str(self.box)).getPosition()).copy()

            # always some y and z-offset because of the way the wedge and the boxes were placed
            opt = box_pos.copy()
            opt[2] -= 0.3
            # additional offset in x-direction and y-direction dependent on skill
            # (which side do we want to push box from?)
            opt[0] += self.offset * self.opt_pos_dir[self.skill, 0]
            opt[1] += self.offset * self.opt_pos_dir[self.skill, 1]

            loc = self.scene.C.getJointState()[:3]  # current location

            # reward: max distance - current distance
            reward += 0.1 * (self.max_dist - np.linalg.norm(opt - loc)) / self.max_dist

            dist, _ = self.scene.C.eval(ry.FS.distance, ["box" + str(self.box), "wedge"])
            reward += 0.1 * dist[0]

            # give additional reward for pushing puzzle piece into correct direction
            # line from start to goal goes only in x-direction for this skill
            reward += (box_pos[0] - self.box_init[0]) / (self.box_goal[0] - self.box_init[0])

        # optionally give reward of one when box was successfully pushed to other field
        if self.reward_on_change:
            if not (self.init_sym_state == self.scene.sym_state).all():
                print("SYM STATE CHANGED!!!")
                reward += 50

        return reward
