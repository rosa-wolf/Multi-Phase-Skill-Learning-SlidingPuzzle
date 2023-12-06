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
                 path='slidingPuzzle_small.g',
                 seed=12345,
                 num_skills=8,
                 snapRatio=4.,
                 max_steps=100,
                 evaluate=False,
                 random_init_pos=True,
                 random_init_config=True,
                 random_init_board=False,
                 penalize=False,
                 puzzlesize = [2, 2],
                 give_sym_obs=False,
                 sparse_reward=False,
                 reward_on_change=False,
                 reward_on_end=False,
                 term_on_change=False,
                 setback=False,
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
        :param reward_on_change:    whether to give additional reward when box is successfully pushed (default false)
        :param term_on_change:      whether to terminate episode on change of symbolic observation (default false)
        :param z_cov                inverse cov of gaussian like function, for calculating optimal z position dependent on
                                    current x-and y-position
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

        # parameters to control initial env configuration
        self.random_init_pos = random_init_pos
        self.random_init_config = random_init_config
        self.random_init_board = random_init_board

        # parameters to control different versions of observation and reward
        self.give_sym_obs = give_sym_obs
        self.sparse_reward = sparse_reward
        self.reward_on_change = reward_on_change
        self.reward_on_end = reward_on_end

        # is skill execution possible?
        self.skill_possible = None

        # if we only give reward on last episode then we should terminate on change of symbolic observation
        if self.reward_on_end:
            self.term_on_change = True
        else:
            self.term_on_change = term_on_change

        # evaluation mode (if true terminate scene on change of symbolic observation)
        # for evaluating policy on a visual level
        self.evaluate = evaluate

        # should the puzzle board be setback to initial configuration on change of symbolic state
        self.setback = setback

        # has actor fulfilled criteria of termination
        self.terminated = False
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

       ## load fully trained forward model
       #self.fm = ForwardModel(batch_size=60, learning_rate=0.001)
       #self.fm.model.load_state_dict(torch.load("../SEADS_SlidingPuzzle/forwardmodel/models/best_model"))
       #self.fm.model.eval()

        # reset to make sure that skill execution is possible after env initialization
        #self.reset()

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

        # before resetting store symbolic state (forward model needs this information)
        info = self.scene.sym_state.copy()

        # get reward (if we don't only want to give reward on last step of episode)
        reward = self._reward()

        # check if symbolic observation changed
        if not (self._old_sym_obs == self.scene.sym_state).all():
            # for episode termination on change of symbolic observation
            if self.term_on_change or self.evaluate:
                self.terminated = True
            else:
                # only do setback for sparse reward, because for move-reward it penalizes last correct action, that pushes
                # puzzle piece onto neighboring field
                if self.setback:
                    # setback to previous state to continue training until step limit is reached
                    # make sure reward and obs is calculated before this state change
                    # first set actor out of the way,
                    # and after puzzle piece was reset set actor back to its current position
                    self.scene.q = [self.scene.q[0], self.scene.q[1], 0., self.scene.q[3]]

                    self.scene.sym_state = self._old_sym_obs
                    self.scene.set_to_symbolic_state()

        # look whether conditions for termination are met
        # make sure to reset env in trainings loop if done
        done = self._termination()
        if not done:
            self.env_step_counter += 1
        else:
            # if we want to always give a reward on the last episode, even if the symbolic observation did not change
            if self.reward_on_end:
                # TODO: give a positive reward if skill was not applicable, otherwise give zero/negative reward
                if not self.skill_possible:
                    # only get the reward for not moving the block if skill execution was not possible
                    if (self.init_sym_state == self.scene.sym_state).all():
                        # give positive reward for not changing symbolic state if skill execution is not possible
                        reward += 5

            # reset if terminated
            # Caution: make sure this does not change the returned values
            #self.reset()

        # caution: dont return resetted values but values that let to reset
        return obs, reward, done, info

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None, skill=None) -> tuple[dict[str, Any], dict[Any, Any]]:
        """
        Resets the environment (including the agent) to the initial conditions.
        """
        super().reset(seed=seed)
        self.scene.reset()
        self.terminated = False
        self.skill_possible = True
        self.env_step_counter = 0
        self.episode += 1

        # sample skill
        if skill is None:
            self.skill = np.random.randint(0, self.num_skills, 1)[0]
        else:
            self.skill = skill

        # ensure that orientation of actor is such that skill execution is possible
        # skills where orientation of end-effector does not have to be changed for
        no_orient_change = [1, 3, 4, 6]
        if self.skill in no_orient_change:
            self.scene.q0[3] = 0.
        else:
            self.scene.q0[3] = np.pi / 2.

        if self.random_init_pos:
            # Set agent to random initial position inside a box
            init_pos = np.random.uniform(-0.25, .25, (2,))
            self.scene.q = [init_pos[0], init_pos[1], self.scene.q0[2], self.scene.q0[3]]
        if self.random_init_config:
            # TODO: what is with the case we do not have a random initial configuration?
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
            self.scene.set_to_symbolic_state(hard=True)
            self.init_sym_state = sym_obs.copy()

        # look which box is in the field we want to push from
        # important for reward shaping
        field = self.skills[self.skill][0]
        # get box that is currently on that field
        self.box = np.where(self.scene.sym_state[:, field] == 1)[0]

        # look whether skill execution is possible
        # look whether there is a puzzle piece in the field we want to push from
        if self.box.shape == (1,):
            # check if there is a puzzle piece in the field we want to push to
            if np.sum(self.init_sym_state[:, self.skills[self.skill, 1]]) > 0:
                print("field we want to push to is already occupied")
                self.skill_possible = False
            else:
                self.box = self.box[0]

                curr_pos = (self.scene.C.getFrame("box" + str(self.box)).getPosition()).copy()
                max_pos = np.array([0.25, 0.25, 0.25]) * np.concatenate((self.max[self.skill], np.array([1])))
                self.max_dist = np.linalg.norm(curr_pos - max_pos)

                # set init and goal position of box
                self.box_init = self.scene.discrete_pos[self.skills[self.skill, 0]]
                self.box_goal = self.scene.discrete_pos[self.skills[self.skill, 1]]

                # calculate goal sym_state
                self.goal_sym_state = self.init_sym_state.copy()
                # box we want to push should move to field we want to push to
                self.goal_sym_state[self.box, self.skills[self.skill, 0]] = 0
                self.goal_sym_state[self.box, self.skills[self.skill, 1]] = 1

        else:
            print("Skill execution not possible")
            self.skill_possible = False


        self._old_sym_obs = self.scene.sym_state.copy()

        return self._get_observation()

    def seed(self, seed=None):
        np.random.seed(seed)
        #self.np_random, seed = seeding.np_random(seed)
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
        q, _, _ = self.scene.state
        obs = q[:3]

        ## not only add position of relevant puzzle piece, but of all puzzle pieces
        #for i in range(self.num_pieces):
        #    obs = np.concatenate((obs, (self.scene.C.getFrame("box" + str(i)).getPosition()).copy()))

        # do not arrange them in the order of box-idx, but in order of field-idx
        # e.g. if piece 3 is on field 0 it goes first in the observation
        # go through symbolic state and get idx of box that is in each field
        ## TODO: How do I encode unoccupied field
        #for i in range(self.scene.sym_state.shape[1]):
        #    box_idx = np.where(self.scene.sym_state[:, i] == 1)[0]
        #    if box_idx.shape == (0, ):
        #        # if there is no box in that field
        #        obs = np.concatenate((obs, np.array([10., 10., 10.])))
        #    else:
        #        obs = np.concatenate((obs, (self.scene.C.getFrame("box" + str(box_idx[0])).getPosition()).copy()))

        # only take coordinates of relevant puzzle piece for now
        obs = np.concatenate((obs, (self.scene.C.getFrame("box" + str(self.box)).getPosition()).copy()))


        # add executed skill to obervation/state (as one-hot encoding)
        one_hot = np.zeros(shape=self.num_skills)
        one_hot[self.skill] = 1
        obs = np.concatenate((obs, one_hot))

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

        # add dimensions for position of all (relevant) puzzle pieces (x, y, z -position)
        shape += 3 # (self.num_pieces + 1) * 3

        # add space needed for one-hot encoding of skill
        shape += self.num_skills

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
        action[:2] = action[:2] / 3.33
        #action[2] = action[2] / 5.714 - 0.075  # if limits are [-.25, .1]
        action[2] = action[2] / 8. - 0.125  # if limits are [-0.25, 0.]

        for i in range(100):
            # get current position
            act = self.scene.q[:3]

            diff = action - act
            diff *= 2

            self.scene.v = np.array([diff[0], diff[1], diff[2] * 3, 0.])
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
        if self.skill_possible:
            if not self.sparse_reward:
                # read out position of box that should be pushed
                box_pos = (self.scene.C.getFrame("box" + str(self.box)).getPosition()).copy()

                # always some y and z-offset because of the way the wedge and the boxes were placed
                opt = box_pos.copy()
                opt[2] -= 0.31
                #opt[1] -= self.offset / 2
                # additional offset in x-direction and y-direction dependent on skill
                # (which side do we want to push box from?)
                opt[0] += self.offset * self.opt_pos_dir[self.skill, 0]
                opt[1] += self.offset * self.opt_pos_dir[self.skill, 1]

                #max = np.concatenate((self.max[self.skill], np.array([0.25])))
                loc = self.scene.C.getJointState()[:3]  # current location

                # reward: max distance - current distance
                #reward += 0.2 * (self.max_dist - np.linalg.norm(opt - loc)) / self.max_dist

                # give additional reward for pushing puzzle piece into correct direction
                # line from start to goal goes only in x-direction for this skill
                # TODO replace index for those boxes that should be pushed in y-direction
                y_dir = [1, 3, 4, 6]
                if self.skill in y_dir:
                    idx = 1
                else:
                    idx = 0
                reward += (box_pos[idx] - self.box_init[idx]) / (self.box_goal[idx] - self.box_init[idx])

                # minimal negative distance between box and actor
                dist, _ = self.scene.C.eval(ry.FS.distance, ["box" + str(self.box), "wedge"])
                reward += -0.2 * dist[0]

            # optionally give reward of one when box was successfully pushed to other field
            if self.reward_on_change:
                # give this reward every time we are in goal symbolic state
                # not only when we change to it (such that it is markovian)
                if (self.scene.sym_state == self.goal_sym_state).all():
                    # play sound for debugging
                    #playsound("/home/rosa/Documents/Uni/Masterarbeit/SEADS_SlidingPuzzle/Debugging/buzzer-dog-39284.mp3")
                    # only get reward for moving the block, if that was the intention of the skill
                    if self.skill_possible:
                        reward += 1
                    if self.sparse_reward:
                        reward += 49

        return reward
