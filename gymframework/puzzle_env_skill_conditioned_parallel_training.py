from typing import Optional, Tuple, Any, Dict

import gymnasium as gym
import numpy as np
import time
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
from gymnasium.utils import seeding

from puzzle_scene_new_ordering import PuzzleScene
#from robotic import ry
import robotic as ry
import torch
from scipy import optimize
import logging as lg
from get_neighbors import get_neighbors

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
                 lookup=False,
                 snapRatio=4.,
                 fm_path=None,
                 train_fm=True,
                 second_best=True,
                 num_skills=2,
                 skill=0,
                 max_steps=100,
                 puzzlesize = [2, 2],
                 sparse_reward=False,
                 reward_on_change=False,
                 reward_on_end=False,
                 logging=True,
                 term_on_change=False,
                 relabel=False,
                 seed=12345,
                 verbose=0):

        """
        Args:
        :param path: path to scene file
        :param max_steps: Maximum number of steps per episode
        :param sparse_reward:       whether to only give a reward on change of symbolic observation (default false)
        :param reward_on_change:    whether to give additional reward when boxes is successfully pushed (default false)
        :param term_on_change:      whether to terminate episode on change of symbolic observation (default false)
        :param verbose:      _       whether to render scene (default false)
        """
        self.seed(seed)
        self.num_skills = num_skills

        self.second_best = second_best

        # which policy are we currently training? (Influences reward)
        self.skill = skill

        self.num_pieces = puzzlesize[0] * puzzlesize[1] - 1

        # parameters to control different versions of observation and reward
        self.sparse_reward = sparse_reward
        self.reward_on_change = reward_on_change
        self.reward_on_end = reward_on_end

        self.neighborlist = self.__get_neighbors(puzzle_size=puzzlesize)

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

        # store rewards of all skills for whole episode (for relabeling)
        self.episode_rewards = []

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
        self.end_epis = False

        # list of boxes to push for each skill
        self.boxes = None

        # initially dummy environment (we will not train this, so learning parameters are irrelevant)
        # only used for loading saved fm into
        self.lookup = lookup
        self.train_fm = train_fm
        self.fm_path = fm_path
        self.logging = logging

        if self.lookup:
            from forwardmodel_lookup.forward_model import ForwardModel
        else:
            from forwardmodel_simple_input.forward_model import ForwardModel

        self.fm = ForwardModel(num_skills=self.num_skills, puzzle_size=puzzlesize)

        if not self.train_fm:
            if self.lookup:
                raise "Loading pretrained model not yet implemented for lookup table"
            if self.fm_path is None:
                raise ValueError("No path for pretrained forward model given")
            self.fm.model.load_state_dict(torch.load(fm_path))
            self.starting_epis = False
        else:
            if self.lookup:
                print("initial save of fm")
                self.fm.save(fm_path)
            else:
                print("initial save of fm")
                torch.save(self.fm.model.state_dict(), fm_path)

    @ staticmethod
    def __get_neighbors(puzzle_size):
        return get_neighbors(puzzle_size)

    def step(self, action: Dict) -> tuple[
        dict[str, np.ndarray | Any], float, bool, bool, dict[str, np.ndarray[int] | None | Any]]:
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
            if self.logging:
                lg.info(f"Change with skill {self.skill} after {self.total_num_steps} steps")

        # get reward (if we don't only want to give reward on last step of episode)
        if self.relabel:
            rewards = self._get_rewards_for_all_skills()
            #print("rewards = ", rewards)
            self.episode_rewards.append(rewards)

            reward = rewards[self.skill]
        else:
            reward = self._reward()

        max_rewards = None
        max_skill_one_hot = None
        success = 0.
        if self._termination():
            #print(f"init_sym_state =\n {self.init_sym_state},\n out_sym_state = \n{self.scene.sym_state}")
            #print(f"relabel = {self.relabel}")
            if self.relabel:
                #max_skill, max_reward = self.get_max_skill_reward(reward)

                # get skill where return is maximized
                self.episode_rewards = np.array(self.episode_rewards)
                max_skill = np.argmax(np.sum(self.episode_rewards, axis=0))
                max_rewards = self.episode_rewards[:, max_skill]

                max_skill_one_hot = np.zeros(self.num_skills)
                max_skill_one_hot[max_skill] = 1
                max_skill_one_hot = max_skill_one_hot[None, :]
                #print(max_rewards)
                max_rewards = max_rewards[:, None]

                #print(f"max_skill = {max_skill}")
                #print(f"rewards = {max_rewards}")

                #print(f"max_skill = {max_skill}")

            # calculate whether episode was success
            success = self.fm.calculate_reward(self.fm.sym_state_to_input(self.init_sym_state.flatten()),
                                               self.fm.sym_state_to_input(self.scene.sym_state.flatten()),
                                               self.skill,
                                               normalize=False,
                                               log=False)

        else:
            self.env_step_counter += 1
            self.total_num_steps += 1

        #print(f"reward = {reward}")
        return (obs,
                reward,
                self.terminated,
                self.truncated,
                {"max_rewards": max_rewards, "max_skill": max_skill_one_hot, 'is_success': success})

    def _get_box(self, k):
        """
        Looks up which boxes should be pushed by the given skill, based on the forward model
        :param k: skill as scalar
        :return: boxes idx, or -1 if skill is not executable in given field
        """
        skill = np.zeros((self.num_skills,))
        skill[k] = 1

        empty_out, _ = self.fm.successor(self.fm.sym_state_to_input(self.init_sym_state), skill, sym_output=False)
        # get the boxes that is currently on the empty_out field
        empty_out = np.where(empty_out == 1)[0]

        box = np.where(self.init_sym_state[:, empty_out[0]] == 1)[0]
        if box.shape == (0,):
            box = -1
            print("no box should be pushed")
        else:
            box = box[0]
            print(f"box {box} should be pushed")

        return box


    def reset(self,
              *,
              seed: Optional[int] = None,
              skill=None,
              actor_pos=None,
              sym_state_in=None,
              options: Optional[dict] = None, ) -> tuple[dict[str, Any], dict[Any, Any]]:
        """
        Resets the environment (including the agent) to the initial conditions.
        """
        super().reset(seed=seed)
        #self.scene.reset()
        self.terminated = False
        self.truncated = False
        self.episode_rewards = []
        self.env_step_counter = 0
        self.episode += 1

        # sample skill
        if skill is not None:
            self.skill = skill
        else:
            if not self.end_epis:
                self.skill = np.random.randint(0, self.num_skills, 1)[0]
            else:
                # at end do not sample from all skills, but only from those the forward model predicts change for
                # (for those that are in use)
                # out is shape empty_fields x num_skill x output array
                used_skills = set()
                out = self.fm.get_full_pred()
                for i in range(out.shape[0]):
                    out[i, :, i] = 0.
                    used_skills.update(list(np.where(np.any(out[i] > 0.5, axis=1))[0]))
                used_skills = np.array(list(used_skills))
                if used_skills.shape == (0,):
                    raise ValueError("No skills lead to any change in symbolic state, "
                                     "reward scheme seems to have been changed to early")
                self.skill = np.random.choice(np.array(list(used_skills)))

        print(f"skill = {self.skill}")
        #print("skill = ", self.skill)
        # orientation of end-effector is always same
        self.scene.q0[3] = np.pi / 2.

        if actor_pos is None:
            # Set agent to random initial position inside a boxes
            init_pos = np.random.uniform(-0.25, .25, (2,))
            self.scene.q = [init_pos[0], init_pos[1], self.scene.q0[2], self.scene.q0[3]]
            #print("current pos = ", self.scene.q)
        else:
            self.scene.q = [actor_pos[0], actor_pos[1], self.scene.q0[2], self.scene.q0[3]]

        if sym_state_in is None:
            # randomly pick the field where no block is initially
            field = np.delete(np.arange(0, self.scene.pieces + 1),
                              np.random.choice(np.arange(0, self.scene.pieces + 1)))
            # put blocks in random fields, except the one that has to be free
            order = np.random.permutation(field)
            sym_obs = np.zeros((self.scene.pieces, self.scene.pieces + 1))
            for i in range(self.scene.pieces):
                sym_obs[i, order[i]] = 1
        else:
            sym_obs = sym_state_in

        self.scene.sym_state = sym_obs
        self.scene.set_to_symbolic_state()
        self.init_sym_state = sym_obs.copy()

        #print(f"init_sym_obs = {self.init_sym_state}")

        # if we have a path to a forward model given, that means we are training the fm and policy in parallel
        # we have to reload the current forward model
        if self.train_fm:
            if self.lookup:
                self.fm.load(self.fm_path)
            #print("Reloading fm")
            else:
                self.fm.model.load_state_dict(torch.load(self.fm_path))

        self._old_sym_obs = self.scene.sym_state.copy()

        # get boxes forward model predicts we should push for each skill
        self.boxes = [self._get_box(skill) for skill in np.arange(self.num_skills)]

        #print(f"boxes = {self.boxes}")

        return self._get_observation(), {}

    def execution_reset(self, skill):
        """
        Reset for when we execute a solution path. Here we do not want the puzzle board to reset
        We only set the actor pos to the initial z-plane over its current position, set the new skill, and get an observation
        :return: observation after the reset
        """
        self.terminated = False
        self.truncated = False
        self.episode_rewards = []
        self.env_step_counter = 0

        actor_pos = np.array([self.scene.q[0], self.scene.q[1], self.scene.q0[2]])
        self._goto_pos(actor_pos)

        self.init_sym_state = self.scene.sym_state

        self.skill = skill

        return self._get_observation()

    def _goto_pos(self, goal_pos):

        act = self.scene.q[:3]
        diff = goal_pos - act
        max_vel = np.array([diff[0], diff[1], diff[2], 0.])

        #self.scene.q = np.array([goal_pos[0], goal_pos[1], self.scene.q0[2], self.scene.q0[3]])
        #return 0

        while not np.allclose(np.linalg.norm(goal_pos - self.scene.q[:3]), 0., atol=0.05):
            #print(f"diff = {goal_pos - act}")
            #print(f"factor = {np.linalg.norm(diff)}")
            self.scene.v = 10 * np.array([diff[0], diff[1], diff[2], 0.])
            self.scene.velocity_control(1)

            act = self.scene.q[:3]
            diff = goal_pos - act



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

        action[0] /= 4
        action[1] /= 4

        action[2] = action[2] / 10 - 0.1
        for _ in range(50):
            # get current position
            act = self.scene.q[:3]

            diff = action - act

            self.scene.v = 2 * np.array([diff[0], diff[1], diff[2], 0.])
            self.scene.velocity_control(1)

    def _get_neg_dist_to_neighbors(self, empty_field):
        """
        Get the negative distance to all boxes on the adjacent fields to the empty one
        :param empty_field: the field that is initially empty (as a scalar)

        :returns: the neg min distance
        """
        min_dist = -np.inf
        neighbor_fields = self.neighborlist[str(empty_field)]
        neighbor_pieces = []

        # get boxes on adjacent fields from sym state
        for field in neighbor_fields:
            box = np.where(self.scene.sym_state[:, field] == 1)[0][0]
            neighbor_pieces.append(box)

        for box in neighbor_pieces:
            dist, _ = self.scene.C.eval(ry.FS.distance, ["box" + str(box), "wedge"])
            if dist[0] > min_dist:
                min_dist = dist[0]

        return min([min_dist, 0.])

    def _contact_with_nonneighbor(self, empty_field) -> bool:
        """
        Returns true if wedge is in contact with box, that is not adjacent to an empty field
        """

        neighbor_fields = self.neighborlist[str(empty_field)]
        neighbor_pieces = []

        # get boxes on adjacent fields from sym state
        for field in neighbor_fields:
            box = np.where(self.scene.sym_state[:, field] == 1)[0][0]
            neighbor_pieces.append(box)

        for box in range(self.num_pieces):
            if box not in neighbor_pieces:
                dist, _ = self.scene.C.eval(ry.FS.distance, ["box" + str(box), "wedge"])
                if dist >= 0:
                    return True, box

        return False, None

    def _get_rewards_for_all_skills(self):
        """
        Calculates rewards for all skills
        :return: list of rewards sorted by skill-enumeration
        """
        return [self._reward(skill) for skill in np.arange(self.num_skills)]

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
            reward += min([2 * self.fm.novelty_bonus(self.fm.sym_state_to_input(self.init_sym_state.flatten()),
                                                self.fm.sym_state_to_input(self.scene.sym_state.flatten()),
                                                k), 2.])
            #print(f"novelty reward = {reward}")

        if self._termination():
            print("terminating")
            # if we want to always give a reward on the last episode, even if the symbolic observation did not change
            if self.reward_on_end:
                if not self.starting_epis:
                    take_max = np.max(
                        [-2., 2 * self.fm.calculate_reward(self.fm.sym_state_to_input(self._old_sym_obs.flatten()),
                                                       self.fm.sym_state_to_input(self.scene.sym_state.flatten()),
                                                       k, second_best=self.second_best)])

                    end_reward = min([take_max, 2.])
                    reward += end_reward
                    print(f"end reward = {end_reward}")


        if not self.sparse_reward:
            if not self.end_epis:
                # penalize being away from all fields where the adjacent field is empty
                # and penalize contact with puzzle piece, that cannot be pushed
                # get empty field
                empty = np.where(np.sum(self.scene.sym_state, axis=0) == 0)[0][0]
                min_dist = self._get_neg_dist_to_neighbors(empty)
                #print(min_dist)
                reward += 0.5 * min_dist
                #print(f"neg dist reward to all neighbors = {reward}")

                ## only penalize contact with wrong boxes, if we are not in contact with correct ones
                #if min_dist < -0.01:
                #    contact, box = self._contact_with_nonneighbor(empty)
                #    if contact:
                #        reward -= 0.1
                #        #print(f"In contact with box {box}")


            else:
                if self.boxes[k] != -1:
                    dist, _ = self.scene.C.eval(ry.FS.distance, ["box" + str(self.boxes[k]), "wedge"])
                    #print(f"min dist to single box = {dist[0]}")
                    reward += 0.5 * min([dist[0], 0.])

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
        #print("reward = ", reward)

        return reward

    def _get_max_skill_reward(self, reward):
        """
        Returns the skill and reward, that gives the maxmimal reward for a given transition
        """
        #print("---------------------------\nCalculating new skill and reward for relabeling")
        max_reward = reward
        skill = self.skill
        print(f"init skill = {skill}, init_reward = {max_reward}")
        for k in range(self.num_skills):
            if k != self.skill:
                reward = self._reward(k=k)
                #print(f"skill = {k}, reward = {reward}")
                if reward > max_reward:
                    max_reward = reward
                    skill = k

        print(f"max skill = {skill}, max_reward = {max_reward}\n-----------------------------------------")

        one_hot_skill = np.zeros(shape=self.num_skills, dtype=np.int8)
        one_hot_skill[skill] = 1
        max_reward = np.array(max_reward, dtype=np.float32)

        return one_hot_skill, max_reward