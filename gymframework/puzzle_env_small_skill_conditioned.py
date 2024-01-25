from typing import Optional, Tuple, Dict, Any

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
                 path='slidingPuzzle_small.g',
                 snapRatio=4.,
                 fm_path=None,
                 skill=0,
                 max_steps=100,
                 random_init_pos=True,
                 random_init_config=True,
                 random_init_board=False,
                 penalize=False,
                 puzzlesize = [1, 2],
                 give_sym_obs=False,
                 sparse_reward=False,
                 reward_on_change=False,
                 reward_on_end=False,
                 term_on_change=False,
                 seed=12345,
                 verbose=0):

        """
        Args:
        :param skill: which skill we want to train (influences reward and field configuration)
        :param path: path to scene file
        :param max_steps: Maximum number of steps per episode
        :param random_init_pos:     whether agent should be placed in random initial position on start of episode (default true)
        :param random_init_config:  whether puzzle pieces should be in random positions initially (default true)
        :param random_init_board:   whether to NOT ensure that skill execution is possible in initial board configuration (default false)
        :param give_sym_obs:        whether to give symbolic observation in agents observation (default false)
        :param sparse_reward:       whether to only give a reward on change of symbolic observation (default false)
        :param reward_on_change:    whether to give additional reward when box is successfully pushed (default false)
        :param term_on_change:      whether to terminate episode on change of symbolic observation (default false)
                                    current x-and y-position
        :param verbose:      _       whether to render scene (default false)
        """
        self.seed(seed=seed)
        # ground truth skills
        # we have only one box, so there is only one skill
        self.skills = np.array([[1, 0], [0, 1]])
        self.num_skills = 2

        # which policy are we currently training? (Influences reward)
        self.skill = skill

        self.num_pieces = puzzlesize[0] * puzzlesize[1] - 1

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
        self.opt_pos_dir = np.array([[1, 0], [-1, 0]])
        # store which box we will push with current skill
        self.box = None

        # TODO: we cannot hardcode a position where the actor will have the maximal distance to the optimal position
        # as the optimal position changes with the position of the box
        # However, we can hardcode a maximal distance using the
        self.max = np.array([[-1, 1], [1, 1]])
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

        # set init and goal position of box for calculating reward
        self.box_init = None
        self.box_goal = None

        self.init_sym_state = None

        ## load fully trained forward model
        #self.fm = ForwardModel(width=2,
        #                       height=1,
        #                       num_skills=2,
        #                       batch_size=10,
        #                       learning_rate=0.001)
#
        self.fm_path = fm_path
        #if self.fm_path is None:
        #    self.fm.model.load_state_dict(
        #        torch.load("/home/rosa/Documents/Uni/Masterarbeit/SEADS_SlidingPuzzle/forwardmodel_simple_input/models/best_model_change"))
        #else:
        #    # load forward model we currently train
        #    self.fm.model.load_state_dict(torch.load(self.fm_path))
        #self.fm.model.eval()
        ## reset to make sure that skill execution is possible after env initialization
        ##self.reset()

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
        current_sym_state = self.scene.sym_state.copy()

        reward = 0
        # get reward (if we don't only want to give reward on last step of episode)
        reward = self._reward()

        # check if symbolic observation changed
        if self.term_on_change:
            self.terminated = not (self._old_sym_obs == self.scene.sym_state).all()

        # look whether conditions for termination are met
        # make sure to reset env in trainings loop if done
        done = self._termination()
        if not done:
            self.env_step_counter += 1
        else:
            # if we want to always give a reward on the last episode, even if the symbolic observation did not change
            if self.reward_on_end:
                # TODO: give a positive reward if skill was not applicable, otherwise give zero/negative reward
                #if not self.skill_possible:
                #    # only get the reward for not moving the block if skill execution was not possible
                #    if (self.init_sym_state == self.scene.sym_state).all():
                #        # give positive reward for not changing symbolic state if skill execution is not possible
                #        # reward += 5
                #        pass

                # give reward calculated by forward model, in a well trained fm this should only be positive if skill
                # execution was no possible
                # only add reward if we did not yet do this when calculating reward
                if (self._old_sym_obs == self.scene.sym_state).all():
                    reward += self.fm.calculate_reward(self.fm.sym_state_to_input(self._old_sym_obs.flatten()),
                                                       self.fm.sym_state_to_input(self.scene.sym_state.flatten()),
                                                       self.skill)

                print("reward_on_end")

        return (obs, reward,
                self.terminated,
                self.truncated,
                {"init_sym_state": self.init_sym_state, "sym_state": current_sym_state, "skill": self.skill})

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
        self.skill_possible = True
        self.env_step_counter = 0
        self.episode += 1

        # sample skill
        self.skill = np.random.randint(0, self.num_skills, 1)[0]

        # ensure that orientation of actor is such that skill execution is possible
        # skills where orientation of end-effector does not have to be changed for
        #no_orient_change = [1, 4, 6, 7, 9, 12]
        #if self.skill in no_orient_change:
        #    self.scene.q0[3] = 0.
        #else:
        self.scene.q0[3] = np.pi / 2.

        if self.random_init_pos:
            # Set agent to random initial position inside a box
            init_pos = np.random.uniform(-0.25, .25, (2,))
            self.scene.q = [init_pos[0], init_pos[1], self.scene.q0[2], self.scene.q0[3]]
        if self.random_init_config:
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
            self.init_sym_state = sym_obs.copy()

        # look which box is in the field we want to push from
        # important for reward shaping
        field = self.skills[self.skill][0]
        # get box that is currently on that field
        self.box = np.where(self.scene.sym_state[:, field] == 1)[0]

        # TODO: if there is not box in that field skill is not applicable
        # TODO: (in 1x2 case that is the only case that matters, as there is only one box)
        # TODO: In bigger fields we also have to look at the case that there already is a box in the field we
        # TODO: want to push to
        # look whether skill execution is possible:
        if self.box.shape == (1,):
            self.box = self.box[0]

            curr_pos = (self.scene.C.getFrame("box" + str(self.box)).getPosition()).copy()
            max_pos = np.array([0.25, 0.25, 0.25]) * np.concatenate((self.max[self.skill], np.array([1])))
            self.max_dist = np.linalg.norm(curr_pos - max_pos)

            # set init and goal position of box
            self.box_init = (self.scene.C.getFrame("box" + str(self.box)).getPosition()).copy()
            self.box_goal = self.scene.discrete_pos[self.skills[self.skill, 1]]
        else:
            print("Skill execution not possible")
            self.skill_possible = False

        # if we have a path to a forward model given, that means we are training the fm and policy in parallel
        # we have to reload the current forward model
        if self.fm_path is not None:
            self.fm.model.load_state_dict(torch.load(self.fm_path))


        self._old_sym_obs = self.scene.sym_state.copy()

        print("skill = ", self.skill)

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

    def _get_observation(self):
        """
        Returns the observation:    Robot joint states and velocites and symbolic observation                                      Executed Skill is also encoded in observation/state
        """
        q, _, _ = self.scene.state
        q = q[:3] * 4.
        q = q.astype(np.float32)

        # one-hot encoding of initially empty field
        empty = np.where(np.sum(self.init_sym_state, axis=0) == 0)[0][0]
        one_hot_empty = np.zeros((self.num_pieces + 1,), dtype=np.float32)
        one_hot_empty[empty] = 1

        # coordinates of boxes on the fields, with encoding whether field is empty (1) or occupied (0)
        pos = np.empty((self.num_pieces + 1, 2), dtype=np.float32)
        curr_empty = np.zeros((self.num_pieces + 1,), dtype=np.float32)
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
        one_hot_skill = np.zeros(shape=self.num_skills, dtype=np.float32)
        one_hot_skill[self.skill] = 1

        # print("obs = ", np.concatenate((q, one_hot_empty, curr_empty, pos, one_hot_skill)))

        return np.concatenate((q, one_hot_empty, curr_empty, pos, one_hot_skill))

    @property
    def observation_space(self):
        """
        Defines bounds of the observation space (Hard coded for now)
        """
        # observation space as 1D array instead
        # joint configuration (3) + skill (1)
        shape = 3  # 5 #+ self.scene.sym_state.shape[0] * self.scene.sym_state.shape[1]

        # for init empty field
        shape += self.num_pieces + 1
        # for current empty field
        shape += self.num_pieces + 1

        # add dimensions for position of all (relevant) puzzle pieces (x, y-position)
        shape += (self.num_pieces + 1) * 2

        # add space needed for one-hot encoding of skill
        shape += self.num_skills

        return Box(low=-1., high=1., shape=(shape,), dtype=np.float64)

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
        reward = 0
        if self.skill_possible:
            if not self.sparse_reward:
                # read out position of box that should be pushed
                box_pos = (self.scene.C.getFrame("box" + str(self.box)).getPosition()).copy()

                # always some y and z-offset because of the way the wedge and the boxes were placed
                opt = box_pos.copy()
                opt[2] -= 0.3
                opt[1] -= self.offset / 2
                # additional offset in x-direction and y-direction dependent on skill
                # (which side do we want to push box from?)
                opt[0] += self.offset * self.opt_pos_dir[self.skill, 0]
                opt[1] += self.offset * self.opt_pos_dir[self.skill, 1]

                #max = np.concatenate((self.max[self.skill], np.array([0.25])))
                loc = self.scene.C.getJointState()[:3]  # current location

                # reward: max distance - current distance
                reward += 0.1 * (self.max_dist - np.linalg.norm(opt - loc)) / self.max_dist

                # give neg distance reward
                dist, _ = self.scene.C.eval(ry.FS.distance, ["box" + str(self.box), "wedge"])
                reward += 0.1 * dist[0]

                # give additional reward for pushing puzzle piece into correct direction
                # line from start to goal goes only in x-direction for this skill
                reward += (box_pos[0] - self.box_init[0]) / (self.box_goal[0] - self.box_init[0])


            # optionally give reward of one when box was successfully pushed to other field
            if self.reward_on_change:
                # give this reward every time we are in goal symbolic state
                # not only when we change to it (such that it is markovian))
                if not (self.init_sym_state == self.scene.sym_state).all():
                    # only get reward for moving the block, if that was the intention of the skill
                    #reward += 50 * self.fm.calculate_reward(self.fm.sym_state_to_input(self._old_sym_obs.flatten()),
                    #                                       self.fm.sym_state_to_input(self.scene.sym_state.flatten()),
                    #                                       self.skill)

                    # add a constant reward to encourage state change early on in training
                    reward += 1
                    print("SYM STATE CHANGED!!!")
        else:
            if self.reward_on_change:
                if not (self.init_sym_state == self.scene.sym_state).all():
                    # only get reward for moving the block, if that was the intention of the skill
                    # reward += 50 * self.fm.calculate_reward(self.fm.sym_state_to_input(self._old_sym_obs.flatten()),
                    #                                       self.fm.sym_state_to_input(self.scene.sym_state.flatten()),
                    #                                       self.skill)

                    # add a constant reward to encourage state change early on in training
                    reward -= 1
                    print("SYM STATE CHANGED BUT SHOULD NOT HAVE")

        print("reward = ", reward)

        return reward

    def relabeling(self, episodes):
        """
        Relabeling of the episoded such that the number of times a certain skill k is applied in the relabeled episoded
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
        row_idx, _ = optimize.linear_sum_assignment(prob, maximize=True)

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
            new_skill = skill_array[row_idx[i]]


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
                    reward = prob[i, row_idx[i]] + np.log(self.num_skills)
                    if not (fm_epi[0] == fm_epi[2]).all():
                        # if state changed in the episode give a larger reward
                        reward *= 50
                        # add a constant reward to encourage state change early on in training
                        reward += 10

                    rewards[-1] = reward

                    # add new states to episode
                    epi = (tuple(states_0), epi[1], rewards, tuple(states_1), epi[4])
                    # get it back into old shape
                    epi = tuple(zip(*epi))

                rl_episodes.append(epi)

        return rl_episodes, fm_episodes




