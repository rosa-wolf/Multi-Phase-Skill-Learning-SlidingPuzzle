from typing import Optional, Tuple, Dict, Any

import gym
import numpy as np
import time
from gym.core import ObsType, ActType
from gym.spaces import Box, Dict, Discrete, MultiBinary
from gym.utils import seeding
from puzzle_scene import PuzzleScene
from robotic import ry


class PuzzleEnv(gym.Env):
    """
    Custom environment for the sliding puzzle
    """

    def __init__(self,
                 skill: int,
                 path='slidingPuzzle.g',
                 max_steps=30,
                 random_init_pos=True,
                 random_init_config=True,
                 verbose=0,
                 nsubsteps=1):

        """
        Args:
        :param skill: which skill we want to train (influences reward and field configuration)
        :param max_steps: Maximum number of steps per episode
        :param random_init_pos: whether agent should be placed in random initial position on start of episode
        """

        # has actor fullfilled criteria of termination
        self.terminated = False
        self.env_step_counter = 0
        self._max_episode_steps = max_steps
        self.episode = 0

        self.scene = PuzzleScene(path, verbose=verbose)

        self.dt = self.scene.tau

        self.random_init_pos = random_init_pos
        self.random_init_config = random_init_config

        self.skill = skill

        # first two is velocity in x-y plane, last decides wether we perform skill (>0) or not (<=0)
        self.action_space = Box(low=np.array([-2., -2., 0]), high=np.array([2., 2., 1]), shape=(3,), dtype=np.float64)

        self._old_sym_obs = self.scene.sym_state.copy()

        # variable to remember weather robot was setback because it went outside of boundary
        self._setback = False

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

        # store preivious symbolic observation
        self._old_sym_obs = self.scene.sym_state.copy()

        # apply action
        self.apply_action(action)

        # check if joints are outside observation_space
        # if so set back to last valid position
        if not self.scene.check_limits():
            # go back to previous position (orientation cannot change currently)
            self._setback = True

        # get observation
        #self._old_obs = self._obs.copy()
        #self._old_sym_obs = self._sym_obs.copy()

        self._obs = self._get_observation()
        # copy observation to return it unchanged
        obs = self._obs.copy()

        # get reward
        reward = self._reward()

        # for episode termination on change of symbolic observation
        # check if symbolic observation changed
        if not (self._old_sym_obs == self.scene.sym_state).all():
            #    self.terminated = True
            self.scene.sym_state = self._old_sym_obs
            self.scene.set_to_symbolic_state()

        # look whether conditions for termination are met
        done = self._termination()
        if not done:
            self.env_step_counter += 1
            # set puzzle pieces back to discrete places (in case they were moved just a little bit
            # but not enough do change symbolic observation)
            self.scene.set_to_symbolic_state()
        else:
            # reset env
            self.reset()

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
            # assumes that skills are enumerated as in file gather-transitions
            # get random intiial board configuration where skill execution is possible
            if self.skill <= 1:
                field = np.array([1, 2, 3, 4, 5])
            elif self.skill <= 4:
                field = np.array([0, 2, 3, 4, 5])
            elif self.skill <= 6:
                field = np.array([0, 1, 3, 4, 5])
            elif self.skill <= 8:
                field = np.array([0, 1, 2, 4, 5])
            elif self.skill <= 11:
                field = np.array([0, 1, 2, 3, 5])
            elif self.skill <= 13:
                field = np.array([0, 1, 2, 3, 4])

            # put blocks in random fields, except the one that has to be free
            order = np.random.permutation(field)
            sym_obs = np.zeros((5, 6))
            for i in range(5):
                sym_obs[i, order[i]] = 1

            self.scene.sym_state = sym_obs
            self.scene.set_to_symbolic_state()

        self._old_sym_obs = self.scene.sym_state.copy()
        self._obs = self._get_observation()

        return self._obs

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
            #print("Termination: Environment reset now")
            # no reset yet because then we never get obsrvation that let to termination
            #self.reset()

            #self.terminated = False
            #self.env_step_counter = 0

            return True
        return False

    def _get_observation(self):
        """
        Returns the observation: Robot joint states and velocites and symbolic observation
        """
        q, q_dot, sym_obs = self.scene.state
        return q[:2] # np.concatenate((q[:3]))#, q_dot[:2]))#, sym_obs.flatten()))
        #return {"q": np.array(q)[:3], "q_dot": np.array(q_dot)[:2], "sym_obs": sym_obs}

    @property
    def observation_space(self):
        """
        Defines bounds of the observation space (Hard coded for now)
        """
        # Todo: implement reading out actual limits from file
        # observation space as 1D array instead
        shape = 2 #5 #+ self.scene.sym_state.shape[0] * self.scene.sym_state.shape[1]
        # make observation spae one single array (such that it works with sac algorithm)
        # 0-2 joint position in x,y,z
        # 3, 4: velocity in x,y direction
        # 5 - end: symbolic observation (flattened)
        return Box(low=-np.inf, high=np.inf, shape=(shape,), dtype=np.float64)

    def apply_action(self, action):
        """
        Applys the action in the scene
        :param action: 2D velocity, and [-1, 1] indicating whether we want to  execute skill afterward
        :return:
        """
        #
        # read out action (velocities in x-y plane)
        #self.scene.v = np.array([action["q_dot"][0], action["q_dot"][1], 0., 0.])
        self.scene.v = np.array([action[0], action[1], 0., 0.])
        # apply action (for now always for (just) ten steps)
        self.scene.velocity_control(15)
        # execute skill if action says so
        #if action["perform_skill"] == 1:
        #    self.execute_skill()

        if action[2] >= 0.5:
            # check if executing action can even lead to wanted change in symbolic observation to save time in training
            # format of limit [[xmin, ymin], [xmax, ymax]]
            if self.skill == 0:
                lim = np.array([[0.02, -0.18], [0.15, 0.02]])
            elif self.skill == 1:
                lim = np.array([[-0.23, 0.05], [-0.05, 0.2]])
            elif self.skill == 2:
                lim = np.array([[-0.5, -0.2], [-0.13, 0.02]])
            elif self.skill == 3:
                lim = np.array([[0.13, -0.2], [0.5, 0.02]])
            elif self.skill == 4:
                lim = np.array([[-0.17, 0.05], [-0.05, 0.2]])
            elif self.skill == 5:
                lim = np.array([[-0.15, -0.18], [-0.02, 0.02]])
            elif self.skill == 6:
                lim = np.array([[0.05, 0.05], [0.23, 0.2]])
            elif self.skill == 7:
                lim = np.array([[-0.23, -0.2], [-0.05, -0.05]])
            elif self.skill == 8:
                lim = np.array([[0.02, -0.02], [0.15, 0.18]])
            elif self.skill == 9:
                lim = np.array([[-0.17, -0.2], [-0.05, -0.05]])
            elif self.skill == 10:
                lim = np.array([[-0.5, -0.02], [-0.13, 0.2]])
            elif self.skill == 11:
                lim = np.array([[0.13, -0.02], [0.5, 0.2]])
            elif self.skill == 12:
                lim = np.array([[0.05, -0.2], [0.23, -0.05]])
            elif self.skill == 13:
                lim = np.array([[-0.15, -0.02], [-0.02, 0.18]])


            if (lim[0, :] <= self.scene.q[:2]).all() and (self.scene.q[:2] <= lim[1, :]).all():
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

        # set robot back to position before skill execution (such that reward will be calculated correctly)
        self.scene.C.setJointState(prev_joint_pos)
        self.scene.S.setState(self.scene.C.getFrameState())
        self.scene.S.step([], self.scene.tau, ry.ControlMode.none)

    def _reward(self) -> float:
        """
        Calculates reward, which is based on symbolic observation change
        """
        # TODO: reward shaping
        # for sparse reward
        #reward = 0

        ## reward positive, linear and higher the smaller the distance to optimal point is
        # use when fixed episode number
        # define place of highest reward for each skill
        if self.skill == 0:
            opt = np.array([0.0559, -0.07, self.scene.q0[2], self.scene.q0[3]])
        elif self.skill == 1:
            opt = np.array([-0.11, 0.11, self.scene.q0[2], self.scene.q0[3]])
        elif self.skill == 2:
            opt = np.array([-0.17, -0.07, self.scene.q0[2], self.scene.q0[3]])
        elif self.skill == 3:
            opt = np.array([0.17, -0.07, self.scene.q0[2], self.scene.q0[3]])
        elif self.skill == 4:
            opt = np.array([0., 0.11, self.scene.q0[2], self.scene.q0[3]])
        elif self.skill == 5:
            opt = np.array([-0.0559, -0.07, self.scene.q0[2], self.scene.q0[3]])
        elif self.skill == 6:
            opt = np.array([0.11, 0.11, self.scene.q0[2], self.scene.q0[3]])
        elif self.skill == 7:
            opt = np.array([-0.11, -0.11, self.scene.q0[2], self.scene.q0[3]])
        elif self.skill == 8:
            opt = np.array([0.0559, 0.07, self.scene.q0[2], self.scene.q0[3]])
        elif self.skill == 9:
            opt = np.array([0., -0.11, self.scene.q0[2], self.scene.q0[3]])
        elif self.skill == 10:
            opt = np.array([-0.17, 0.07, self.scene.q0[2], self.scene.q0[3]])
        elif self.skill == 11:
            opt = np.array([0.17, 0.07, self.scene.q0[2], self.scene.q0[3]])
        elif self.skill == 12:
            opt = np.array([0.11, -0.11, self.scene.q0[2], self.scene.q0[3]])
        elif self.skill == 13:
            opt = np.array([-0.0559, 0.07, self.scene.q0[2], self.scene.q0[3]])

        max = np.array([-0.2, 0.2, self.scene.q0[2], self.scene.q0[3]])#  location with the lowest reward (lower right corner)
        loc = self.scene.C.getJointState()  # current location

        # reward: max distance - current distance
        reward = np.linalg.norm(opt - max) - np.linalg.norm(opt - loc)

        ## negative reward that considers angle and distance
        #opt = np.array([0.11114188, -0.09727765, self.scene.q0[2], self.scene.q0[3]])  # loc with the highest reward (reward = 0)
        #loc = self.scene.C.getJointState()  # current location
        ## distance to opt
        #reward = - np.linalg.norm(opt - loc)
        ## if y smaller than that of location take sin (else take times 1)
        #if loc[1] < opt[1]:
        #    h = np.array([opt[0], loc[1], self.scene.q0[2], self.scene.q0[3]])  # helper point to calculate sin of angle
        #    reward *= np.linalg.norm(loc - h) / np.linalg.norm(loc - opt)


        # extra reward if symbolic observation changed
        if not (self._old_sym_obs == self.scene.sym_state).all():
            reward += 1


        return reward

        #######################################################################
        ## set reward zero except to last transition in an episode
        # if self.terminated:
        #    # make reward dependent on distance to location where skill should be executed
        #    loc = np.array([0.11114188, -0.09727765, 0.,  0.])
        #    # lower right corner should have max dist to that position
        #    max_dist = np.linalg.norm(loc - np.array([-0.2, 0.2, 0., 0.]))
        #
        #    # current position
        #    pos = self.scene.C.getJointState()
        #    dist = np.linalg.norm(loc - pos)
        #
        #    return 1 - np.log(1 + dist)/np.log(1 + max_dist)
        #
        # return 0
