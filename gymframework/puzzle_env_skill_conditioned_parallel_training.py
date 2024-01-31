from typing import Optional, Tuple, Any

import gymnasium as gym
import numpy as np
import time
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
from gymnasium.utils import seeding
from puzzle_scene_new_ordering import PuzzleScene
from robotic import ry
import torch
from scipy import optimize
import logging

from forwardmodel_simple_input.forward_model import ForwardModel


class PuzzleEnv(gym.Env):
    """
    Custom environment for the sliding puzzle

    This is an environment to train the skill-conditioned policy.
    The skill is set during every reset of the environment
    The skill is part of the observation.

    This is the version to test out having a smaller (1x2) board
    """

    def __init__(self,
                 log_dir,
                 path='slidingPuzzle_2x2.g',
                 snapRatio=4.,
                 fm_path=None,
                 train_fm=True,
                 num_skills=2,
                 skill=0,
                 max_steps=100,
                 puzzlesize = [2, 2],
                 sparse_reward=False,
                 reward_on_change=False,
                 reward_on_end=False,
                 term_on_change=False,
                 relabel=False,
                 verbose=0):

        """
        Args:
        :param path: path to scene file
        :param max_steps: Maximum number of steps per episode
        :param sparse_reward:       whether to only give a reward on change of symbolic observation (default false)
        :param reward_on_change:    whether to give additional reward when box is successfully pushed (default false)
        :param term_on_change:      whether to terminate episode on change of symbolic observation (default false)
        :param verbose:      _       whether to render scene (default false)
        """
        # ground truth skills
        # we have only one box, so there are 2 skills
        self.num_skills = num_skills

        # which policy are we currently training? (Influences reward)
        self.skill = skill

        self.num_pieces = puzzlesize[0] * puzzlesize[1] - 1

        # parameters to control different versions of observation and reward
        self.sparse_reward = sparse_reward
        self.reward_on_change = reward_on_change
        self.reward_on_end = reward_on_end

        # is skill execution possible?
        self.skill_possible = None

        self.neighborlist = self._get_neighbors(puzzlesize)

        # if we only give reward on last episode then we should terminate on change of symbolic observation
        if self.reward_on_end:
            self.term_on_change = True
        else:
            self.term_on_change = term_on_change

        # only do relabeling for sparse reward
        if not self.term_on_change:
            self.relabel = False
        else:
            self.relabel = relabel

        # has actor fulfilled criteria of termination
        self.terminated = False
        self.truncated = False
        self.env_step_counter = 0
        self.total_num_steps = 0
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

        # if true we are in the starting episodes where the environment may reward differently
        self.starting_epis = True

        self.box = None

        # initially dummy environment (we will not train this, so learning parameters are irrelevant)
        # only used for loading saved fm into
        self.fm = ForwardModel(width=puzzlesize[1],
                               height=puzzlesize[0],
                               num_skills=self.num_skills)

        self.train_fm = train_fm
        self.fm_path = fm_path
        if not self.train_fm:
            if self.fm_path is None:
                raise ValueError("No path for pretrained forward model given")
            self.fm.model.load_state_dict(torch.load(fm_path))
            self.starting_epis = False
        else:
            print("initial save of fm")
            torch.save(self.fm.model.state_dict(), fm_path)


        logging.basicConfig(filename=log_dir + '/change.log', level=logging.INFO, filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s')

    def _get_neighbors(self, puzzle_size):
        """
        Calculates the neibhors of each field, depending on the puzzle size
        """
        neighborlist = {}
        for i in range(puzzle_size[0]):
            for j in range(puzzle_size[1]):
                # look on which side we have a neighor and mark it in array
                neighbors = []
                field = j + (i * puzzle_size[1])
                if i != 0:
                    # a neighbor above
                    neighbors.append(field - puzzle_size[1])
                if j != 0:
                    # no neighbor on left
                    neighbors.append(field - 1)
                if j != puzzle_size[1] - 1:
                    # a neighbors on right
                    neighbors.append(field + 1)
                if i != puzzle_size[0] - 1:
                    # a neighbor below
                    neighbors.append(field + puzzle_size[1])

                neighborlist[str(j + (i * puzzle_size[1]))] = neighbors

        return neighborlist

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

        # check if symbolic observation changed
        if not (self._old_sym_obs == self.scene.sym_state).all():
            # for episode termination on change of symbolic observation
            self.terminated = self.term_on_change
            logging.info(f"Change with skill {self.skill} after {self.total_num_steps} steps")

        # get reward (if we don't only want to give reward on last step of episode)
        reward = self._reward()

        max_reward = None
        max_skill = None
        if self._termination():
            #print(f"relabel = {self.relabel}")
            if self.relabel:
                max_skill, max_reward = self._get_max_skill_reward(reward)
                #print(f"max_skill = {max_skill}")
        else:
            self.env_step_counter += 1
            self.total_num_steps += 1

        return (obs,
                reward,
                self.terminated,
                self.truncated,
                {"max_reward": max_reward, "max_skill": max_skill})

    def reset(self,
              *,
              seed: Optional[int] = None,
              skill=None,
              options: Optional[dict] = None, ) -> tuple[dict[str, Any], dict[Any, Any]]:
        """
        Resets the environment (including the agent) to the initial conditions.
        """
        super().reset(seed=seed)
        self.scene.reset()
        self.terminated = False
        self.truncated = False
        self.skill_possible = True
        self.env_step_counter = 0
        self.episode += 1

        # sample skill
        if skill is not None:
            self.skill = skill
        else:
            self.skill = np.random.randint(0, self.num_skills, 1)[0]
        #print("skill = ", self.skill)
        # orientation of end-effector is always same
        self.scene.q0[3] = np.pi / 2.

        # Set agent to random initial position inside a box
        init_pos = np.random.uniform(-0.25, .25, (2,))
        self.scene.q = [init_pos[0], init_pos[1], self.scene.q0[2], self.scene.q0[3]]
        #print("current pos = ", self.scene.q)

        # randomly pick the field where no block is initially
        field = np.delete(np.arange(0, self.scene.pieces + 1),
                          np.random.choice(np.arange(0, self.scene.pieces + 1)))
        # put blocks in random fields, except the one that has to be free
        order = np.random.permutation(field)
        sym_obs = np.zeros((self.scene.pieces, self.scene.pieces + 1))
        for i in range(self.scene.pieces):
            sym_obs[i, order[i]] = 1

        self.scene.sym_state = sym_obs
        self.scene.set_to_symbolic_state()
        self.init_sym_state = sym_obs.copy()

        #print(f"init_sym_obs = {self.init_sym_state}")

        # if we have a path to a forward model given, that means we are training the fm and policy in parallel
        # we have to reload the current forward model
        if self.train_fm:
            #print("Reloading fm")
            self.fm.model.load_state_dict(torch.load(self.fm_path))

        self._old_sym_obs = self.scene.sym_state.copy()

        # get box forward model predicts we should push
        # penalize being far away from box fm predicts we should push
        skill = np.zeros((self.num_skills,))
        skill[self.skill] = 1
        empty_out = self.fm.successor(self.init_sym_state, skill, sym_output=False)
        # get the box that is currently on the empty_out field
        empty_out = np.where(empty_out == 1)[0]

        self.box = np.where(self.init_sym_state[:, empty_out[0]] == 1)[0]
        if self.box.shape == (0,):
            self.skill_possible = False
            self.box = None
        else:
            self.box = self.box[0]

        return self._get_observation(), {}

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
        # actor joint cooridinates
        q, _, _ = self.scene.state
        q = q[:3] * 4  # for normilization
        q = q.astype(np.float32)

        # one-hot encoding of initially empty field
        empty = np.where(np.sum(self.init_sym_state, axis=0) == 0)[0][0]
        one_hot_empty = np.zeros((self.num_pieces + 1,), dtype=np.int8)
        one_hot_empty[empty] = 1

        # coordinates of boxes on the fields, with encoding whether field is empty (1) or occupied (0)
        pos = np.empty((self.num_pieces + 1, 2), dtype=np.float32)
        curr_empty = np.zeros((self.num_pieces + 1,), dtype=np.int8)
        for i in range(self.scene.sym_state.shape[1]):
            box_idx = np.where(self.scene.sym_state[:, i] == 1)[0]
            if box_idx.shape[0] == 0:
                curr_empty[i] = 1  # encode whether field is empty
                pos[i] = self.scene.discrete_pos[i, :2] * 4
            else:
                box_idx = box_idx[0]
                pos[i] = (self.scene.C.getFrame("box" + str(box_idx)).getPosition()[:2] * 4).copy()

        pos = pos.flatten()

        # one-hot encoding of skill
        one_hot_skill = np.zeros(shape=self.num_skills, dtype=np.int8)
        one_hot_skill[self.skill] = 1

        obs = {"q": q,
               "init_empty": one_hot_empty,
               "curr_empty": curr_empty,
               "box_pos": pos,
               "skill": one_hot_skill}

        return obs

    @property
    def observation_space(self):
        """
        Defines bounds and shape of the observation space
        """
        obs_space = {"q": Box(low=-1., high=1., shape=(3,)),
                     "init_empty": MultiBinary(self.num_pieces + 1), # Box(low=-1, high=1, shape=(self.num_pieces + 1,)),
                     "curr_empty": MultiBinary(self.num_pieces + 1),
                     "box_pos": Box(low=-1., high=1., shape=((self.num_pieces + 1) * 2,)),
                     "skill": MultiBinary(self.num_skills)}  # Box(low=-1, high=1, shape=(self.num_skills,))}

        return Dict(obs_space)

    def apply_action(self, action):
        """
        Applys the action in the scene
        :param action: desired x-y-z position
        """
        # do velocity control for 100 steps
        for _ in range(100):
            # get current position
            act = self.scene.q[:3]

            diff = action / 4 - act

            self.scene.v = np.array([diff[0], diff[1], diff[2], 0.])
            self.scene.velocity_control(1)

    def _reward(self, k=None) -> float:
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
        reward = 0.

        if k is None:
            k = self.skill

        if not (self.init_sym_state == self.scene.sym_state).all():
            # always give novelty bonus when state changes
            print("SYM STATE CHANGED !!!")
            # add novelty bonus (min + 0)
            reward += 5 * self.fm.novelty_bonus(self.fm.sym_state_to_input(self.init_sym_state.flatten()),
                                                self.fm.sym_state_to_input(self.scene.sym_state.flatten()),
                                                k)

        if self._termination():
            print("terminating")
            # if we want to always give a reward on the last episode, even if the symbolic observation did not change
            if self.reward_on_end:
                if not self.starting_epis:
                    reward += np.max(
                        [-1., self.fm.calculate_reward(self.fm.sym_state_to_input(self._old_sym_obs.flatten()),
                                                       self.fm.sym_state_to_input(self.scene.sym_state.flatten()),
                                                       k)])

                    print("reward_on_end = ", reward)

        if not self.sparse_reward:
            if self.starting_epis:
                # penalize being away from all fields where the adjacent field is empty
                min_dist = -np.inf
                # get empty field
                empty = np.where(np.sum(self.scene.sym_state, axis=0) == 0)[0][0]
                neighbors = self.neighborlist[str(empty)]
                for i in range(len(neighbors)):
                    dist, _ = self.scene.C.eval(ry.FS.distance, ["box" + str(i), "wedge"])
                    if dist[0] > min_dist:
                        min_dist = dist[0]

                reward += 5 * min([min_dist, 0.])
            else:
                if self.skill_possible:
                    dist, _ = self.scene.C.eval(ry.FS.distance, ["box" + str(self.box), "wedge"])
                    reward += 5 * min([dist[0], 0.])

       #else:
       #    if not self.sparse_reward:
       #        # give a small reward calculated by the forward model in every step
       #        reward += 0.001 * self.fm.calculate_reward(self.fm.sym_state_to_input(self._old_sym_obs.flatten()),
       #                                                   self.fm.sym_state_to_input(
       #                                                   self.scene.sym_state.flatten()),
       #                                                   k)
       #    if self.reward_on_change:
       #        if not self.reward_on_end:
       #            # give this reward every time we are in goal symbolic state
       #            # not only when we change to it (such that it is markovian))
       #            if not (self.init_sym_state == self.scene.sym_state).all():
       #                reward += np.max(
       #                    [-1., self.fm.calculate_reward(self.fm.sym_state_to_input(self._old_sym_obs.flatten()),
       #                                                   self.fm.sym_state_to_input(self.scene.sym_state.flatten()),
       #                                                   k)])
        print("reward = ", reward)

        return reward

    def _get_max_skill_reward(self, reward):
        """
        Returns the skill and reward, that gives the maxmimal reward for a given transition
        """
        #print("---------------------------\nCalculating new skill and reward for relabeling")
        max_reward = reward
        skill = self.skill
        #print(f"init skill = {skill}, init_reward = {max_reward}")
        for k in range(self.num_skills):
            if k != self.skill:
                reward = self._reward(k=k)
                #print(f"skill = {k}, reward = {reward}")
                if reward > max_reward:
                    max_reward = reward
                    skill = k

        #print(f"max skill = {skill}, max_reward = {max_reward}\n-----------------------------------------")

        one_hot_skill = np.zeros(shape=self.num_skills, dtype=np.int8)
        one_hot_skill[skill] = 1
        max_reward = np.array(max_reward, dtype=np.float32)

        return one_hot_skill, max_reward

    def relabeling(self, episodes):
        """
        Relabeling of the episodes such that the number of times a certain skill k is applied in the relabeled episoded
        is equal to the number of times it was applied in the input transitions.
        This is done, to maintain the property of the skills being equally likely
        (which is assumed in the derivation of the reward)
        For this to work the number of input epsiodes n has to be sufficiently large

        This function solves the linear sum assignment problem to get the new skills k for the episodes

        @param episodes: list of n episodes in the form
                ((s_0, a_0, r_0, s_1, m_0), (s_1, a_1, r_1, s_2, m_1), ..., (s_T-1, a_T-1, r_T-1, s_T, m_T), (e_0, k, e_T))
                , the state s contains a one-hot encoding of the skill that needs to be relabeled,
                e_0 and e_T are one-hot encodings of the empty fields
        returns: relabeled episodes in same shape as input episodes
        """

        # count the number of times each skill appears
        count = np.zeros(self.num_skills, dtype=int)
        # go through all episodes and count how often each skill appears
        for epi in episodes:
            # the last tuple in each episode is of the form (z_0, k, z_T), and contains the skill as a one-hot encoding
            one_hot_skill = epi[-1][1]
            skill = np.where(one_hot_skill == 1)[0][0]
            count[skill] += 1

        # set up the cost matrix
        # contains one row per episode and c columns per skill, where c is the number of times the skill
        # is applied in the original episodes
        prob = np.zeros((len(episodes), len(episodes)))

        # go through all episodes i and calculate cost q(k | e_iT, e_i0), for all k
        # add this to matrix for all c times the skill k appears
        for i, epi in enumerate(episodes):
            idx = 0
            for k in range(self.num_skills):
                cost = self.fm.calculate_reward(epi[-1][0], epi[-1][-1], k, normalize=False)
                for c in range(count[k]):
                    prob[i, c + idx] = cost
                idx += count[k]

        # get array of skills where each skill is repeated by its number of occurance
        skill_array = np.empty((len(episodes)))
        idx = 0
        for k in range(self.num_skills):
            for c in range(count[k]):
                skill_array[c + idx] = k
            idx += count[k]

        # solve the linear sum assignment problem maximization, using scipy
        _, col_idx = optimize.linear_sum_assignment(prob, maximize=True)

        # relabel the episodes
        rl_episodes = []
        fm_episodes = []
        for i, epi in enumerate(episodes):
            # for each episode take all states and replace the one-hot encoding
            # of the skill with one-hot encoding of the new skill
            # get fm transition
            fm_epi = epi[-1]
            # delete fm transition from episodes
            epi = epi[: -1]

            # look if new skill is equal to old skill
            old_skill = np.where(fm_epi[1] == 1)[0][0]
            new_skill = skill_array[col_idx[i]]

            if old_skill == new_skill:
                rl_episodes.append(epi)
                fm_episodes.append(fm_epi)
            else:
                # Todo: only relabel rl_episodes with certain probability
                # Todo: also recalculate reward
                # relabel all fm-episodes
                one_hot_new_skill = np.zeros(self.num_skills)

                one_hot_new_skill[int(new_skill)] = 1
                fm_episodes.append((fm_epi[0], one_hot_new_skill, fm_epi[2]))

                # only relabel rl-episodes with certain probability
                if np.random.uniform() >= 0.5:

                    epi = tuple(zip(*epi))
                    states_0 = np.array(epi[0])
                    states_1 = np.array(epi[3])
                    rewards = np.array(epi[2])

                    # for the skill the last num_skill entries of the state are responsible
                    states_0[:, -self.num_skills:] = one_hot_new_skill
                    states_1[:, -self.num_skills:] = one_hot_new_skill

                    # Todo: recalculate reward actor would have gotten if he had executed the new skill
                    # Beware: only works for the sparse reward setting
                    # only look at last transition as actor only gets a reward there
                    # reward = self.fm.calculate_reward(fm_epi[0], fm_epi[2], int(new_skill))
                    reward = prob[i, col_idx[i]] + np.log(self.num_skills)
                    if not (fm_epi[0] == fm_epi[2]).all():

                        # if state changed in the episode give a larger reward
                        reward *= 50
                        if reward < 0:
                            reward = 0
                        # add a constant reward to encourage state change early on in training

                    rewards[-1] = reward

                    # add new states to episode
                    epi = (tuple(states_0), epi[1], rewards, tuple(states_1), epi[4])
                    # get it back into old shape
                    epi = tuple(zip(*epi))

                rl_episodes.append(epi)

        return rl_episodes, fm_episodes




