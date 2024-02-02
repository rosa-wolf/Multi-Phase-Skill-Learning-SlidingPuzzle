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

#from forwardmodel.forward_model import ForwardModel


class PuzzleEnv(gym.Env):
    """
    Custom environment for the sliding puzzle

    This is an environment to train the policy for a single skill.
    When initiating the variable for the skill is set and not changed after.
    This is only so we can use the same environment-type for all skills.
    """

    def __init__(self,
                 path='slidingPuzzle.g',
                 skill=0,
                 max_steps=100,
                 evaluate=False,
                 random_init_pos=True,
                 random_init_config=True,
                 random_init_board=False,
                 penalize=False,
                 puzzlesize = [2, 3],
                 give_sym_obs=False,
                 sparse_reward=False,
                 reward_on_end=False,
                 term_on_change=False,
                 z_cov=12,
                 verbose=0):

        """
        Args:
        :param skill: which skill we want to train (influences reward and field configuration)
        :param path: path to scene file
        :param max_steps: Maximum number of steps per episode
        :param evaluate: evaluation mode (default false)
        :param random_init_pos:     whether agent should be placed in random initial position on start of episode (default true)
        :param random_init_config:  whether puzzle pieces should be in random positions initially (default true)
        :param random_init_board:   whether to NOT ensure that skill execution is possible in initial board configuration (default false)
        :param give_sym_obs:        whether to give symbolic observation in agents observation (default false)
        :param sparse_reward:       whether to only give a reward on change of symbolic observation (default false)
        :param reward_on_end:       whether to only give reward on episode termination (default false)
                                    - only use in combination with term_on_change=True
        :param term_on_change:      whether to terminate episode on change of symbolic observation (default false)
        :param z_cov                inverse cov of gaussian like function, for calculating optimal z position dependent on
                                    current x-and y-position
        :param verbose:             whether to render scene (default false)
        """
        # ground truth skills
        self.skills = np.array([[1, 0], [3, 0], [0, 1], [2, 1], [4, 1], [1, 2], [5, 2],
                                [0, 3], [4, 3], [1, 4], [3, 4], [5, 4], [2, 5], [4, 5]])


        # which policy are we currently training? (Influences reward)
        self.skill = skill

        # opt push position for all 14 skills (for calculating reward)
        # TODO: Make optimal position dependent on current position of block we want to push
        # position when reading out block position is the center of the block
        # depending on skill we have to push from different side on the block
        # optimal position is position at offset into right direction from the center og the block
        # offset half the side length of the block
        self.offset = 0.04
        # direction of offset [x-direction, y-direction]
        # if 0 no offset in that direction
        # -1/1: negative/positive offset in that direction
        self.opt_pos_dir = np.array([[1, 0], [0, 1], [-1, 0], [1, 0], [0, 1], [-1, 0], [0, 1],
                                     [0, -1], [1, 0], [0, -1], [-1, 0], [1, 0], [0, -1], [-1, 0]])
        # store which boxes we will push with current skill
        self.box = None

        # opt push position for all 14 skills (for calculating reward)
        self.opt_pos = np.array([[0.06, -0.09],
                                 [-0.1, 0.11],
                                 [-0.155, -0.06],
                                 [0.18, -0.06],
                                 [0.01, 0.11],
                                 [-0.06, -0.06],
                                 [0.13, 0.11],
                                 [-0.12, -0.11],
                                 [0.06, 0.06],
                                 [0., -0.12],
                                 [-0.155, 0.05],
                                 [0.175, 0.05],
                                 [0.115, -0.12],
                                 [-0.05, 0.06]])

        self.z_cov = z_cov
        # parameters to control initial env configuration
        self.random_init_pos = random_init_pos
        self.random_init_config = random_init_config
        self.random_init_board = random_init_board

        # parameters to control different versions of observation and reward
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
        self.env_step_counter = 0
        self._max_episode_steps = max_steps
        self.episode = 0

        # initialize scene
        self.scene = PuzzleScene(path, puzzlesize=puzzlesize, verbose=verbose)

        # desired x-y-z coordinates of eef
        self.action_space = Box(low=np.array([-1., -1., -1.]), high=np.array([1., 1., 1.]), shape=(3,),
                                dtype=np.float64)

        # store symbolic observation from previous step to check for change in symbolic observation
        # and for calculating reward based on forward model
        self._old_sym_obs = self.scene.sym_state.copy()

        # store initial and goal position of boxes for reward shaping
        self.box_init = None
        self.box_goal = None

       ## load fully trained forward model
       #self.fm = ForwardModel(batch_size=60, learning_rate=0.001)
       #self.fm.model.load_state_dict(torch.load("../SEADS_SlidingPuzzle/forwardmodel/models/best_model"))
       #self.fm.model.eval()

        # reset to make sure that skill execution is possible after env initialization
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

        # store current symbolic observation before executing action
        self._old_sym_obs = self.scene.sym_state.copy()

        # apply action
        self.apply_action(action)

        # check if joints are outside observation_space
        # if so set back to last valid position
        self.scene.check_limits()

        obs = self._get_observation()

        if not self.reward_on_end:
            # get reward (if we don't only want to give reward on last step of episode)
            reward = self._reward()

        # check if symbolic observation changed
        if not (self._old_sym_obs == self.scene.sym_state).all():
            # for episode termination on change of symbolic observation
            if self.term_on_change or self.evaluate:
                self.terminated = True
                if self.reward_on_end:
                    reward = self._reward()
            #else:
                # setback to previous state to continue training until step limit is reached
                # make sure reward and obs is calculated before this state change
                #self.scene.sym_state = self._old_sym_obs
                #self.scene.set_to_symbolic_state()

        # look whether conditions for termination are met
        # make sure to reset env in trainings loop if done
        done = self._termination()
        if not done:
            self.env_step_counter += 1
        else:
            # reset if terminated
            # Caution: make sure this does not change the returned values
            self.reset()

        # caution: dont return resetted values but values that let to reset
        return obs, reward, done, {}

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
        self.env_step_counter = 0
        self.episode += 1

        # ensure that orientation of actor is such that skill execution is possible
        # skills where orientation of end-effector does not have to be changed for
        no_orient_change = [1, 4, 6, 7, 9, 12]
        if self.skill in no_orient_change:
            self.scene.q0[3] = 0.
        else:
            self.scene.q0[3] = np.pi / 2.

        if self.random_init_pos:
            # Set agent to random initial position inside a boxes
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

        # look which boxes is in the field we want to push from
        # important for reward shaping
        field = self.skills[self.skill][0]
        # get boxes that is currently on that field
        self.box = np.where(self.scene.sym_state[:, field] == 1)[0][0]

        # set init and goal position of boxes
        self.box_init = (self.scene.C.getFrame("boxes" + str(self.box)).getPosition()).copy()
        self.box_goal = self.scene.discrete_pos[self.skills[self.skill, 1]]

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
        Returns the observation:    Robot joint states and velocites and symbolic observation
                                    Executed Skill is also encoded in observation/state
        """
        # TODO: Include position of relevant puzzle piece in observation
        q, q_dot, sym_obs = self.scene.state
        obs = q[:3]

        # add position of relevant puzzle piece
        obs = np.concatenate((obs, (self.scene.C.getFrame("boxes" + str(self.box)).getPosition()).copy()))

        if self.give_sym_obs:
            # should agent be informed about symbolic observation?
            obs = np.concatenate((obs, sym_obs.flatten()))

        return obs

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
        shape += 3

        # make observation space one single array (such that it works with sac algorithm)
        # 0-2 joint position in x,y,z
        # 3, 4: velocity in x,y direction
        # 5 - end: symbolic observation (flattened)
        if self.give_sym_obs:
            shape += self.scene.sym_state.shape[0] * self.scene.sym_state.shape[1]

        return Box(low=-np.inf, high=np.inf, shape=(shape,), dtype=np.float64)

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

            self.scene.v = 0.9 * np.array([diff[0], diff[1], diff[2], 0.])
            self.scene.velocity_control(1)

    def _reward(self) -> float:
        """
        Calculates reward, which is based on symbolic observation change
        """
        # for sparse reward
        reward = 0
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
                max = np.array([-0.2, 0.2])
            elif self.skill == 1:
                max = np.array([0.2, -0.2])
            elif self.skill == 2:
                max = np.array([0.2, 0.2])
            elif self.skill == 3:
                max = np.array([-0.2, 0.2])
            elif self.skill == 4:
                max = np.array([0.2, -0.2])
            elif self.skill == 5:
                max = np.array([0.2, 0.2])
            elif self.skill == 6:
                max = np.array([-0.2, -0.2])
            elif self.skill == 7:
                max = np.array([0.2, 0.2])
            elif self.skill == 8:
                max = np.array([-0.2, -0.2])
            elif self.skill == 9:
                max = np.array([0.2, 0.2])
            elif self.skill == 10:
                max = np.array([0.2, -0.2])
            elif self.skill == 11:
                max = np.array([-0.2, -0.2])
            elif self.skill == 12:
                max = np.array([-0.2, 0.2])
            elif self.skill == 13:
                max = np.array([0.2, -0.2])

            # define place of highest reward for each skill (optimal position)
            # as outer edge of current position of block we want to push
            # read out position of boxes that should be pushed
            box_pos = (self.scene.C.getFrame("boxes" + str(self.box)).getPosition()).copy()
            # always some y and z-offset because of the way the wedge and the boxes were placed
            opt = box_pos.copy()
            opt[2] -= 0.3
            opt[1] -= self.offset / 2
            # additional offset in x-direction and y-direction dependent on skill
            # (which side do we want to push boxes from?)
            opt[0] += self.offset * self.opt_pos_dir[self.skill, 0]
            opt[1] += self.offset * self.opt_pos_dir[self.skill, 1]
            # max = np.array([-0.2, 0.2, self.scene.q0[2], self.scene.q0[3]])#  location with the lowest reward (lower right corner)
            max = np.concatenate((max, np.array([0.25])))

            # max = np.array([-0.2, 0.2, self.scene.q0[2], self.scene.q0[3]])#  location with the lowest reward (lower right corner)
            loc = self.scene.C.getJointState()[:3]  # current location

            # reward: max distance - current distance
            # only consider distance in x-y-direction
            # opt = self.opt_pos[self.skill]
            # TODO: try out making curve steeper
            # TODO: try out weighting z-distance lower, such that it becomes more important to get to correct x-y-coordinates
            # reward in [0, 0.1]
            reward += 0.1 * (np.linalg.norm(opt - max) - np.linalg.norm(opt - loc))/np.linalg.norm(opt - max)

            # give reward for puzzle piece being pushed in range [0, 1]
            # does path from init to goal go in x-direction (0) or y-direction (1) for the skill
            idx = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0]

            # reward in [0, 1]
            move_reward = (box_pos[idx[self.skill]] - self.box_init[idx[self.skill]])\
                          /(self.box_goal[idx[self.skill]] - self.box_init[idx[self.skill]])
            print("move_reward = ", move_reward)

            reward += move_reward

            # add reward dependent on z-coordinate
            # optimal z is dependent on current x and y (super-gaussian shape)
            # for now lets say the final z should be -0.2
            #cov_inv = np.array([[self.z_cov, 0], [0, self.z_cov]])
            #z_opt_fct = lambda x: -0.2 * np.exp(- (x - opt).T @ cov_inv @ (x - opt))

            # the optimal z has its peak at z = 0 (thus negative and shifted)
            #z_opt = z_opt_fct(loc)

            # give reward dependent on distance of current z to optimal z
            # only positive reward
            #reward += np.linalg.norm(z_opt - 0.25) - np.linalg.norm(z_opt - self.scene.C.getJointState()[2])

        ## extra reward if symbolic observation changed
        if not (self._old_sym_obs == self.scene.sym_state).all():
            # give reward according to forward model
            # TODO: calculate q_k in a different way (which may be more numerically stable)
            #q_k = self.fm.calculate_reward(self._old_sym_obs.flatten(),
            #                               self.scene.sym_state.flatten(),
            #                               self.skill)
            reward += 5

        return reward
