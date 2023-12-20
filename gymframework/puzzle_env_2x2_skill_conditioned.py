from typing import Optional, Tuple, Dict, Any

import gym
import numpy as np
import time
from gym.core import ObsType, ActType
from gym.spaces import Box, Dict, Discrete, MultiBinary
from gym.utils import seeding
from puzzle_scene_new_ordering import PuzzleScene
from robotic import ry
import torch

#from forwardmodel.forward_model import ForwardModel


class PuzzleEnv(gym.Env):
    """
    Custom environment for the sliding puzzle

    This is an environment to train the skill-conditioned policy.
    The skill is set during every reset of the environment
    The skill is part of the observation.

    This is the version to test out having a smaller (1x2) board
    """

    def __init__(self,
                 path='slidingPuzzle_2x2.g',
                 seed=12345,
                 num_skills=8,
                 snapRatio=4.,
                 max_steps=100,
                 puzzlesize = [2, 2],
                 give_sym_obs=False,
                 sparse_reward=False,
                 reward_on_change=False,
                 neg_dist_reward=True,
                 reward_on_end=False,
                 term_on_change=False,
                 verbose=0):

        """
        Args:
        :param path: path to scene file
        :param max_steps: Maximum number of steps per episode
        :param num_skills: How many skills do we want to condition our policy on
        :param random_init_board:   whether to NOT ensure that skill execution is possible in initial board configuration (default false)
        :param sparse_reward:       whether to only give a reward on change of symbolic observation (default false)
        :param reward_on_change:    whether to give additional reward when box is successfully pushed (default false)
        :param term_on_change:      whether to terminate episode on change of symbolic observation (default false)
        :param verbose:      _       whether to render scene (default false)
        """

        # ground truth skills
        # we have only one box, so there is only one skill
        self.skills = np.array([[1, 0], [2, 0],
                                [0, 1], [3, 1],
                                [0, 2], [3, 2],
                                [1, 3], [2, 3]])
        self.num_skills = num_skills

        self.seed(seed=seed)

        # which policy are we currently training? (Influences reward)
        self.skill = None

        self.num_pieces = puzzlesize[0] * puzzlesize[1] - 1

        # opt push position for all 14 skills (for calculating reward)
        # position when reading out block position is the center of the block
        # depending on skill we have to push from different side on the block
        # optimal position is position at offset into right direction from the center og the block
        # offset half the side length of the block
        self.offset = 0.06
        # direction of offset [x-direction, y-direction]
        # if 0 no offset in that direction
        # -1/1: negative/positive offset in that direction
        self.opt_pos_dir = np.array([[-1, 0], [0, 1],
                                     [1, 0], [0, 1],
                                     [0, -1], [-1, 0],
                                     [0, -1], [1, 0]])
        # store which box we will push with current skill
        self.box = None

        # as the optimal position changes with the position of the box
        # However, we can hardcode a maximal distance using the
        self.max = np.array([[1, 1], [-1, -1],
                             [-1, 1], [1, -1],
                             [-1, 1], [1, -1],
                             [1, 1], [-1, -1]])
        self.max_dist = None

        # parameters to control different versions of observation and reward
        self.sparse_reward = sparse_reward
        self.reward_on_change = reward_on_change
        self.reward_on_end = reward_on_end
        self.neg_dist_reward = neg_dist_reward

        # is skill execution possible?
        self.skill_possible = None

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
        self.action_space = Box(low=np.array([-1., -1., -1.]), high=np.array([1., 1., 1.]), shape=(3,),
                                dtype=np.float64)

        # store symbolic observation from previous step to check for change in symbolic observation
        # and for calculating reward based on forward model
        self._old_sym_obs = self.scene.sym_state.copy()

        # set init and goal position of box for calculating reward
        self.box_init = None
        self.box_goal = None

        self.init_sym_state = None
        self.goal_sym_state = None

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
        reward = 0
        # store current symbolic observation before executing action
        self._old_sym_obs = self.scene.sym_state.copy()

        # apply action
        self.apply_action(action)

        # check if joints are outside observation_space
        # if so set back to last valid position
        in_limits = self.scene.check_limits()
        if not in_limits:
            reward -= 0.5

        obs = self._get_observation()

        # get reward (if we don't only want to give reward on last step of episode)
        reward += self._reward()

        # look whether conditions for termination are met
        # make sure to reset env in trainings loop if done
        self._termination()

        self.env_step_counter += 1

        return obs, reward, self.terminated, self.truncated, self.scene.sym_state

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> tuple[dict[str, Any], dict[Any, Any]]:
        """
        Resets the environment (including the agent) to the initial conditions.
        """
        super().reset(seed=seed)
        self.scene.reset()
        self.terminated = False
        self.truncated = False
        self.env_step_counter = 0
        self.episode += 1

        # sample skill
        self.skill = np.random.randint(0, self.num_skills, 1)[0]

        # ensure that orientation of actor is such that skill execution is possible
        # skills where orientation of end-effector does not have to be changed for
        no_orient_change = [1, 3, 4, 6]
        if self.skill in no_orient_change:
            self.scene.q0[3] = 0.
        else:
            self.scene.q0[3] = np.pi / 2.

        # Set agent to random initial position inside a box
        init_pos = np.random.uniform(-0.25, .25, (2,))
        self.scene.q = [init_pos[0], init_pos[1], self.scene.q0[2], self.scene.q0[3]]

        # should it be possible to apply skill on initial board configuration?
        # take initial empty field such that skill execution is possible
        field = np.delete(np.arange(0, self.scene.pieces + 1), self.skills[self.skill, 1])

        # put blocks in random fields, except the one that has to be free
        order = np.random.permutation(field)
        # Todo: reset after adding more puzzle pieces again
        sym_obs = np.zeros((self.scene.pieces, self.scene.pieces + 1))
        for i in range(self.scene.pieces):
            sym_obs[i, order[i]] = 1
        #sym_obs = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        self.scene.sym_state = sym_obs
        self.scene.set_to_symbolic_state(hard=True)
        self.init_sym_state = sym_obs.copy()

        # look which box is in the field we want to push from
        # important for reward shaping
        field = self.skills[self.skill, 0]
        # get box that is currently on that field
        # Todo: set back when adding more pieces again
        self.box = np.where(self.scene.sym_state[:, field] == 1)[0][0]
        #self.box = 0

        curr_pos = (self.scene.C.getFrame("box" + str(self.box)).getPosition()).copy()
        max_pos = np.array([0.25, 0.25, 0.25]) * np.concatenate((self.max[self.skill], np.array([1])))
        self.max_dist = np.linalg.norm(curr_pos - max_pos)

        # set init and goal position of box
        self.box_init = curr_pos #self.scene.discrete_pos[self.skills[self.skill, 0]]
        self.box_goal = self.scene.discrete_pos[self.skills[self.skill, 1]]

        # calculate goal sym_state
        self.goal_sym_state = self.init_sym_state.copy()
        # box we want to push should move to field we want to push to
        self.goal_sym_state[self.box, self.skills[self.skill, 0]] = 0
        self.goal_sym_state[self.box, self.skills[self.skill, 1]] = 1

        self._old_sym_obs = self.scene.sym_state.copy()

        return self._get_observation(), self.init_sym_state

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
            self.truncated = False
            return True
        elif (self.env_step_counter >= self._max_episode_steps - 1):
            self.truncated = True
            return True

        return False

    def _get_observation(self):
        """
        Returns the observation:    Robot joint states and velocites and symbolic observation
                                    Executed Skill is also encoded in observation/state
        """
        q, _, _ = self.scene.state
        obs = q[:3]

        # add coordinates of all puzzle pieces
        for i in range(self.num_pieces):
            obs = np.concatenate((obs, ((self.scene.C.getFrame("box" + str(i)).getPosition()).copy())[:2]))

        # add executed skill to obervation/state (as one-hot encoding)
        one_hot = np.zeros(shape=self.num_skills)
        one_hot[self.skill] = 1
        obs = np.concatenate((obs, one_hot))

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

        # add dimensions for position of all (relevant) puzzle pieces (x, y, z -position)
        shape += self.num_pieces * 2

        # add space needed for one-hot encoding of skill
        shape += self.num_skills

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
        action[:2] /= 4
        action[2] = action[2] / (2/0.3) - 0.05
        # if limits are [-.25, .1]

        for _ in range(100):
            # get current position
            act = self.scene.q[:3]

            diff = action - act

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
        reward = 0
        if not self.sparse_reward:
            # read out position of box that should be pushed
            box_pos = (self.scene.C.getFrame("box" + str(self.box)).getPosition()).copy()

            # always some y and z-offset because of the way the wedge and the boxes were placed
            opt = box_pos.copy()
            opt[2] -= 0.2
            # additional offset in x-direction and y-direction dependent on skill
            # (which side do we want to push box from?)
            opt[0] += self.offset * self.opt_pos_dir[self.skill, 0]
            opt[1] += self.offset * self.opt_pos_dir[self.skill, 1]

            loc = self.scene.C.getJointState()[:3]  # current location

            # reward: max distance - current distance
            #reward += 0.1 * (self.max_dist - np.linalg.norm(opt - loc)) / self.max_dist

            # give additional reward for pushing puzzle piece towards its goal position
            max_dist = np.linalg.norm(self.box_goal - self.box_init)
            box_reward = (max_dist - np.linalg.norm(self.box_goal - box_pos)) / max_dist
            reward += 2 * box_reward
            # minimal negative distance between box and actor
            if self.neg_dist_reward:
                dist, _ = self.scene.C.eval(ry.FS.distance, ["box" + str(self.box), "wedge"])
                reward += 0.1 * dist[0]
            #if np.isclose(dist[0], 0) or dist[0] >= 0:
            #    reward += 0.5
            #    print("give 0.5")


        # optionally give reward of one when box was successfully pushed to other field
        if self.reward_on_change:
            # give this reward every time we are in goal symbolic state
            # not only when we change to it (such that it is markovian)
            if not (self.scene.sym_state == self.init_sym_state).all():
                if (self.scene.sym_state == self.goal_sym_state).all():
                    # only get reward for moving the block, if that was the intention of the skill
                    reward += 3
                else:
                    # punish if wrong block was pushed
                    reward -= 3

                #print("SYM STATE CHANGED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        return reward

    def relabel_all(self, episode):
        """
        Relabels all episodes artificially assuming the skill-effect of the skill, under the assumption that skill
        execution is always possible
        @param: one episode for sac-training (containing multiple transitions), in the form
        ((s_0, a_0, r_0, s_1, m_0), (s_1, a_1, r_1, s_2, m_1), ..., (s_T-1, a_T-1, r_T-1, s_T, m_T), (e_0, k, e_T))

        @returns: relabeled transition for sac-training
        """
        rl_episode = []

        init_sym_state = episode[-1][0]
        end_sym_state = episode[-1][2]

        # if no change in symbolic state happened don't relabel episode
        if (init_sym_state == end_eym_state).all():
            return episode

        # read out which skills corresponds to the change in symbolic state
        empty_1 = np.where(init_sym_state == 1)[0][0]
        empty

        return rl_episode
