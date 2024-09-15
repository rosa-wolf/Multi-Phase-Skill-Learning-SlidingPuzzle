import os
from typing import Optional, Any, Dict
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiBinary
from gymnasium.utils import seeding

import robotic as ry
import torch
import logging as lg

from puzzle_scene import PuzzleScene
from get_neighbors import get_neighbors

class PuzzleEnv(gym.Env):
    """
    Custom environment for the sliding puzzle with a variable puzzle board size

    This is an environment to train the skill-conditioned policy.
    The skill is set during every reset of the environment
    The skill is part of the observation.

    """

    def __init__(self,
                 path='slidingPuzzle_2x2.g',
                 lookup=False,
                 snapRatio=4.,
                 fm_path=None,
                 second_best=True,
                 num_skills=2,
                 skill=0,
                 max_steps=100,
                 puzzlesize = [2, 2],
                 sparse_reward=False,
                 logging=True,
                 init_phase=True,
                 refinement_phase=True,
                 relabel=False,
                 seed=12345,
                 verbose=0):

        """
        Args:
        :param path: path to scene file
        :param lookup: Set to true if the forward model is implemented as a lookup table (bool, default: False)
        :param snapRatio: ratio of distance between cell centers at which symbolic state changes
        :param fm_path: path to the forward model (if we use a pretrained forward model, default: None)
        :param second_best: use second-best normalization in reward
        :param num_skills: number of skills K
        :param skill: skill to use (if we do not want to use randomly sampled one)
        :param max_steps: Maximum number of steps per episode
        :param sparse_reward:       whether to only give a reward on change of symbolic observation (default false)
        :param verbose:      _       whether to render scene (default false)
        """
        self.seed(seed)

        self.num_skills = num_skills
        self.skill = skill

        # puzzle field
        self.num_pieces = puzzlesize[0] * puzzlesize[1] - 1
        self.neighborlist = self.__get_neighbors(puzzle_size=puzzlesize)

        # parameters to control different versions of observation and reward
        self.sparse_reward = sparse_reward
        self.second_best = second_best

        self.relabel = relabel

        # store rewards of all skills for whole episode (for relabeling)
        self.episode_rewards = []

        # has actor fulfilled criteria of termination
        self._max_episode_steps = max_steps
        self.terminated = False
        self.truncated = False
        self.env_step_counter = 0
        self.total_num_steps = 0
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
        self.do_init_phase = init_phase
        self.do_refinement_phase = refinement_phase

        # variables to store in which training phase we currently are
        self.starting_epis = True if self.do_init_phase else False
        self.end_epis = False

        # list of boxes to push for each skill
        self.boxes = None

        # initially dummy fm environment (we will not train this, so learning parameters are irrelevant)
        # only used for loading saved fm into
        self.lookup = lookup
        self.fm_path = fm_path
        self.logging = logging

        if self.lookup:
            from forwardmodel_lookup.forward_model import ForwardModel
        else:
            from forwardmodel_simple_input.forward_model import ForwardModel

        self.fm = ForwardModel(num_skills=self.num_skills, puzzle_size=puzzlesize)

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
            self.terminated = True
            if self.logging:
                lg.info(f"Change with skill {self.skill} after {self.total_num_steps} steps")

        # get reward (if we don't only want to give reward on last step of episode)
        if self.relabel:
            # for relabeling we need to know the rewards for all the skills
            rewards = self._get_rewards_for_all_skills()
            self.episode_rewards.append(rewards)
            reward = rewards[self.skill]
        else:
            reward = self._reward()

        max_rewards = None
        max_skill_one_hot = None
        success = 0.
        if self._termination():
            if self.relabel:
                # get skill where return is maximized
                self.episode_rewards = np.array(self.episode_rewards)
                max_skill = np.argmax(np.sum(self.episode_rewards, axis=0))
                max_rewards = self.episode_rewards[:, max_skill]

                max_skill_one_hot = np.zeros(self.num_skills)
                max_skill_one_hot[max_skill] = 1
                max_skill_one_hot = max_skill_one_hot[None, :]
                max_rewards = max_rewards[:, None]

            # calculate whether episode was success (necessarry to calculate to get success rate)
            success = self.fm.calculate_reward(self.fm.sym_state_to_input(self.init_sym_state.flatten()),
                                               self.fm.sym_state_to_input(self.scene.sym_state.flatten()),
                                               self.skill,
                                               normalize=False,
                                               log=False)

        else:
            self.env_step_counter += 1
            self.total_num_steps += 1

        return (obs,
                reward,
                self.terminated,
                self.truncated,
                {"max_rewards": max_rewards,
                 "max_skill": max_skill_one_hot,
                 "all_rewards": self.episode_rewards,
                 "applied_skill": self.skill,
                 'is_success': success})

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
        else:
            box = box[0]

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
        self.terminated = False
        self.truncated = False
        self.episode_rewards = []
        self.env_step_counter = 0
        self.episode += 1

        if sym_state_in is None:
            sym_obs = self.sample_sym_state()
        else:
            sym_obs = sym_state_in

        self.scene.sym_state = sym_obs
        self.init_sym_state = sym_obs.copy()


        # sample skill
        if skill is not None:
            self.skill = skill
        else:
            if not self.end_epis:
                # uniformly sample skill
                self.skill = np.random.randint(0, self.num_skills, 1)[0]
            else:
                # in refinement phase sample goal field and take skill most probable to reach it
                pred = self.fm.get_pred_for_all_skills(self.fm.sym_state_to_input(self.init_sym_state.flatten()))

                empty_field = np.where(np.sum(self.init_sym_state, axis=0) == 0)[0][0]
                pred[:, empty_field] = 0.

                # sample goal empty field, and take that skill that is most likely to reach it
                goal_field = np.random.choice(np.delete(np.arange(self.num_pieces + 1), empty_field))
                self.skill = np.argmax(pred[:, goal_field])

        # initial actor position and orientation
        self.scene.q0[3] = np.pi / 2.

        if actor_pos is None:
            # Set agent to random initial x-y position inside a boxes on plane parallel to game board
            init_pos = np.random.uniform(-0.25, .25, (2,))
            self.scene.q = [init_pos[0], init_pos[1], self.scene.q0[2], self.scene.q0[3]]
        else:
            self.scene.q = [actor_pos[0], actor_pos[1], self.scene.q0[2], self.scene.q0[3]]

        self.scene.set_to_symbolic_state()

        # if we have a path to a forward model given, that means we are training the fm and policy in parallel
        # we have to reload the current forward model
        if self.lookup:
            self.fm.load(self.fm_path)
        else:
            self.fm.model.load_state_dict(torch.load(self.fm_path))

        self._old_sym_obs = self.scene.sym_state.copy()

        # get boxes forward model predicts we should push for each skill
        self.boxes = [self._get_box(skill) for skill in np.arange(self.num_skills)]

        ###############################
        # get puzzle piece positions
        self.store_box_position = []
        self.store_box_orientation = []
        self.store_q = []
        self.store_sym_state = []
        self.store_skill = []

        box_positions = np.empty((self.num_pieces, 3))
        box_orientations = np.empty((self.num_pieces, 4))
        for i in range(self.num_pieces):
            name = "box" + str(i)
            box_positions[i] = self.scene.C.getFrame(name).getPosition()
            box_orientations[i] = self.scene.C.getFrame(name).getQuaternion()

        self.store_box_position.append(box_positions.flatten())
        self.store_box_orientation.append(box_orientations.flatten())
        self.store_q.append(self.scene.q.flatten())
        self.store_sym_state.append(self.scene.sym_state.flatten())
        self.store_skill.append(self.skill)
        ##############################

        return self._get_observation(), {}

    def sample_sym_state(self):
        # randomly pick the field where no block is initially
        field = np.delete(np.arange(0, self.scene.pieces + 1),
                          np.random.choice(np.arange(0, self.scene.pieces + 1)))
        # put blocks in random fields, except the one that has to be free
        order = np.random.permutation(field)
        sym_obs = np.zeros((self.scene.pieces, self.scene.pieces + 1))
        for i in range(self.scene.pieces):
            sym_obs[i, order[i]] = 1

        return sym_obs

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
        """
        Move actor to specific position using velocity control
        """

        act = self.scene.q[:3]
        diff = goal_pos - act

        while not np.allclose(np.linalg.norm(goal_pos - self.scene.q[:3]), 0., atol=0.05):
            self.scene.v = 10 * np.array([diff[0], diff[1], diff[2], 0.])
            self.scene.velocity_control(1)

            act = self.scene.q[:3]
            diff = goal_pos - act

            ######################################################
            ######################################################
            box_positions = np.empty((self.num_pieces, 3))
            box_orientations = np.empty((self.num_pieces, 4))
            for i in range(self.num_pieces):
                name = "box" + str(i)
                box_positions[i] = self.scene.C.getFrame(name).getPosition()
                box_orientations[i] = self.scene.C.getFrame(name).getQuaternion()

            self.store_box_position.append(box_positions.flatten())
            self.store_box_orientation.append(box_orientations.flatten())
            self.store_q.append(self.scene.q.flatten())
            self.store_sym_state.append(self.scene.sym_state.flatten())
            self.store_skill.append(self.skill)
            ######################################################
            ######################################################

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
                     "init_empty": MultiBinary(self.num_pieces + 1),
                     "curr_empty": MultiBinary(self.num_pieces + 1),
                     "box_pos": Box(low=-1., high=1., shape=((self.num_pieces + 1) * 2,)),
                     "skill": MultiBinary(self.num_skills)}

        return Dict(obs_space)

    def apply_action(self, action):
        """
        Applys the action in the scene
        :param action: desired x-y-z position
        """
        # do velocity control
        action[0] /= 4
        action[1] /= 4

        action[2] = action[2] / 10 - 0.1
        for i in range(50):
            # get current position
            act = self.scene.q[:3]

            diff = action - act

            self.scene.v = 2 * np.array([diff[0], diff[1], diff[2], 0.])
            self.scene.velocity_control(1)
            #if i % 10 == 0:
            #    self.scene.C.view()
            #    self.scene.C.view_savePng('z.vid/')
            ######################################################
            ######################################################
            box_positions = np.empty((self.num_pieces, 3))
            box_orientations = np.empty((self.num_pieces, 4))
            for i in range(self.num_pieces):
                name = "box" + str(i)
                box_positions[i] = self.scene.C.getFrame(name).getPosition()
                box_orientations[i] = self.scene.C.getFrame(name).getQuaternion()

            self.store_box_position.append(box_positions.flatten())
            self.store_box_orientation.append(box_orientations.flatten())
            self.store_q.append(self.scene.q.flatten())
            self.store_sym_state.append(self.scene.sym_state.flatten())
            self.store_skill.append(self.skill)
            ######################################################
            ######################################################

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
            # only add novelty bonus if state changes
            # in init phase give novelty reward (only over other skills)
            # in other phases give novelty bonus (over all skills)
            print("SYM STATE CHANGED !!!")
            reward += self.fm.novelty_bonus(self.fm.sym_state_to_input(self.init_sym_state.flatten()),
                                            self.fm.sym_state_to_input(self.scene.sym_state.flatten()),
                                            k, others_only=self.starting_epis)

        if self._termination():
            # add sparse reward at end of episode
            # if we want to always give a reward on the last episode, even if the symbolic observation did not change
            if not self.starting_epis:
                # clip reward at - log(K)
                reward += np.max(
                    [-np.log(self.num_skills), self.fm.calculate_reward(self.fm.sym_state_to_input(self._old_sym_obs.flatten()),
                                                                        self.fm.sym_state_to_input(self.scene.sym_state.flatten()),
                                                                        k, second_best=self.second_best)])

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

        return reward
