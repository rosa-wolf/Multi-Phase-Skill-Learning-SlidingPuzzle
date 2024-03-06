import os

import gymnasium as gym
import numpy as np
import torch
import logging as lg
import time
from scipy import optimize

from gymnasium.wrappers import TransformReward

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

import os
import sys
dir = os.path.dirname(__file__)
mod_dir = os.path.join(dir, "../../")
sys.path.append(mod_dir)

from FmReplayMemory import FmReplayMemory
from forwardmodel_simple_input.forward_model import ForwardModel

class FmCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, update_freq: int,
                 save_path: str,
                 env,
                 seed: float, memory_size=500,
                 sample_size=40,
                 size=[1, 2],
                 num_skills=2,
                 relabel=False,
                 train_fm=True,
                 logging=False,
                 prior_buffer=False,
                 eval_freq=500,
                 verbose=0):

        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_path = save_path
        self.env = env  # this is not a copy of the env, but a reference to it
        self.sample_size = sample_size
        self.prior_buffer = prior_buffer
        self.relabel = relabel
        self.relabel_buffer = {"max_rewards": [],
                                "max_skill": [],
                               "all_rewards": [],
                               "applied_skill": [],
                                "episode_length": [],
                                "total_num_steps": []}

        # initialize empty replay memory for fm
        self.seed = seed
        self.long_term_buffer = FmReplayMemory(2048, seed)
        self.recent_buffer = FmReplayMemory(256, seed)
        self.verbose = verbose
        self.num_skills = num_skills
        self.train_fm = train_fm
        # initialize forward model (untrained)
        self.fm = ForwardModel(num_skills=self.num_skills, puzzle_size=size, batch_size=sample_size, learning_rate=0.005)

        self.logging = logging
        if self.logging:
            self.test_acc = []
            self.test_loss = []
            self.eval_freq = eval_freq

        # to keep track of circular replay buffer idx
        self.last_done = None

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        # save fm
        if self.verbose > 0:
            print("Saving initial model now")
        # don't save whole model, but only parameters
        # if we train fm save initial model
        if self.train_fm:
            torch.save(self.fm.model.state_dict(), self.save_path + "/fm")

    def _train_fm(self):
        num_updates = int(len(self.relabel_buffer["total_num_steps"]) / 5)
        if num_updates < 1:
            num_updates = 1
        #num_updates = 4
        # make new buffer from recent and sampled long-term episodes
        # only update fm once 256 episodes have been sampled
        if len(self.long_term_buffer) >= 256:
            final_fm_buffer = FmReplayMemory(512, self.seed)
            recent_state_batch, recent_skill_batch, recent_next_state_batch = self.recent_buffer.sample(256)
            state_batch, skill_batch, next_state_batch = self.long_term_buffer.sample(256)

            state_batch = np.concatenate((np.array(recent_state_batch), np.array(state_batch)))
            skill_batch = np.concatenate((np.array(recent_skill_batch), np.array(skill_batch)))
            next_state_batch = np.concatenate((np.array(recent_next_state_batch), np.array(next_state_batch)))

            final_fm_buffer.buffer = list(zip(state_batch, skill_batch, next_state_batch))

            for _ in range(num_updates):
                # put sampling batch(es) from buffer into forward model train function
                train_loss, train_acc = self.fm.train(final_fm_buffer)
                if self.verbose > 0:
                    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
                if self.verbose > 0:
                    print("Saving updated model now")
                # don't save whole model, but only parameters
                torch.save(self.fm.model.state_dict(), self.save_path + "/fm")
    def _eval_fm(self):
        if len(self.buffer) >= self.sample_size:
            if self.verbose > 0:
                print("Evaluate now")
            test_loss, test_acc = self.fm.evaluate(self.buffer)
            self.test_acc.append(test_acc)
            self.test_loss.append(test_loss)
            np.savez(self.save_path + "log_data", test_acc=test_acc, test_loss=test_loss)

            if self.verbose > 0:
                print(f'\tEval Loss: {test_loss:.3f} | Eval Acc: {test_acc * 100:.2f}%')


    def _on_step(self):

        if self.locals["dones"] != 0:
            self.relabel_buffer["max_rewards"].append((self.locals["infos"][0])["max_rewards"])
            self.relabel_buffer["max_skill"].append((self.locals["infos"][0])["max_skill"])
            self.relabel_buffer["all_rewards"].append((self.locals["infos"][0])["all_rewards"])
            self.relabel_buffer["applied_skill"].append((self.locals["infos"][0])["applied_skill"])
            self.relabel_buffer["episode_length"].append(((self.locals["infos"][0])["episode"]["l"]))
            self.relabel_buffer["total_num_steps"].append(self.n_calls)

    def _check_reward_criterion(self) -> bool:
        """
        Checks if reward scheme should be changed and does so if necessarry
        :return: True if scheme has been changed, else False
        """

        # out is shape empty_fields x num_skill x output array
        out = self.fm.get_full_pred()
        change_reward_scheme = True
        for i in range(out.shape[0]):
            # set probabilities to not change empty field to zero, as we are only looking at transitions where change happens
            out[i, :, i] = 0

            # we want for each init empty field at least min_neighbors transitions to other (adjacent) fields being probable
            if self.env.starting_epis:
                # in beginning look whether change regularly happens
                num_change = np.where(out[i] > 0.3)[0].shape[0]
            else:
                # at end look if change happens consistently with high probability
                num_change = np.where(out[i] > 0.5)[0].shape[0]

            if num_change < len(self.env.neighborlist[str(i)]):
                    change_reward_scheme = False

        if change_reward_scheme:
            if self.env.starting_epis:
                self.env.starting_epis = False
                lg.info("Changing from starting reward scheme after {0} steps".format(self.n_calls))
            elif not self.env.end_epis:
                self.env.end_epis = True
                lg.info("Changing to final training part after {0} steps".format(self.n_calls))
            return True

        return False

    def _on_rollout_end(self) -> bool:
        """
        Update fm
        This function is called before the sac networks are updated
        """
        if self.train_fm:
            self._check_reward_criterion()

        # number of episodes to relabel
        #num_relabel = self.locals["num_collected_episodes"] + 1
        num_relabel = len(self.relabel_buffer["total_num_steps"])

        print(f"==================\nnum_relabel = {num_relabel}")

        # for now relabel without caring for skill distribution
        #dones = np.where((self.locals["replay_buffer"]).dones == 1)[0]

        # calculate linear sum assignment problem
        if self.relabel and not self.env.starting_epis:
            self._lin_sum_assignment(num_relabel)

        for i_episode in range(num_relabel):

            #print(f"steps = {self.relabel_buffer['total_num_steps']}, epi length = {self.relabel_buffer['episode_length']}")
            print(f"i_episode = {i_episode}")
            print(f"total_num_steps = {self.relabel_buffer['total_num_steps']}")
            start_idx = self.relabel_buffer["total_num_steps"][i_episode] - self.relabel_buffer["episode_length"][i_episode]
            end_idx = self.relabel_buffer["total_num_steps"][i_episode] - 1

            #print(f"before wrapping: start_idx = {start_idx}, end_idx = {end_idx}")
            start_idx = start_idx % self.locals["replay_buffer"].buffer_size
            end_idx = end_idx % self.locals["replay_buffer"].buffer_size

            # wrap around end of replay buffer
            #print(f"start_idx = {start_idx}, end_idx = {end_idx}")

            init_empty = (self.locals["replay_buffer"]).next_observations["init_empty"][end_idx]
            out_empty = (self.locals["replay_buffer"]).next_observations["curr_empty"][end_idx]
            old_skill = (self.locals["replay_buffer"]).next_observations["skill"][end_idx]
            #print(f"init_empty = {init_empty}, out_empty = {out_empty}")

            if self.relabel and not self.env.starting_epis:
                # get skill that maximizes reward
                # TODO: relabeling for now assumes that we terminated on change of symbolic state
                new_skill = self.relabel_buffer["max_skill"][i_episode]

                #print(f"old_skill = {old_skill}, new_skill = {new_skill}")
                #print(f"start_idx = {start_idx}, end_idx = {end_idx}")
                # never relabel policy transitions
                if not (new_skill == old_skill).all():
                    # relabel policy transitions with 50% probability
                    if np.random.normal() > 0.5:
                        self._relabel(start_idx, end_idx, i_episode, new_skill)

                # always relabel fm transition
                if self.train_fm:
                    self.long_term_buffer.push(init_empty.flatten(), new_skill.flatten(), out_empty.flatten())
                    self.recent_buffer.push(init_empty.flatten(), new_skill.flatten(), out_empty.flatten())
                    #print(f"init = {init_empty.flatten()}, new_skill = {new_skill.flatten()}, old_skill = {old_skill.flatten()}, out_empty = {out_empty.flatten()}")

            else:
                if self.train_fm:
                    self.long_term_buffer.push(init_empty.flatten(), old_skill.flatten(), out_empty.flatten())
                    self.recent_buffer.push(init_empty.flatten(), old_skill.flatten(), out_empty.flatten())

        # train fm after episodes have been added to buffer
        if self.train_fm:
            self._train_fm()
            if self.logging:
                self._eval_fm()


        #print("-------------------------------------------------------")

        self.relabel_buffer = {"max_rewards": [],
                               "max_skill": [],
                               "all_rewards": [],
                               "applied_skill": [],
                               "episode_length": [],
                               "total_num_steps": []}

    def _lin_sum_assignment(self,
                            num_episodes):

        # count the number of times each skill appears
        for k in range(self.num_skills):
            unique, counts = np.unique(self.relabel_buffer["applied_skill"], return_counts=True)

        skill_array = np.zeros(num_episodes)
        # get array of skills where each skill is repeated by its number of occurance
        idx = 0
        for k in range(self.num_skills):
            skill_array[idx: idx + counts[k]] = k
            idx += counts[k]

        # go through all episodes i and calculate cost q(k | e_iT, e_i0), for all k
        # add this to matrix for all c times the skill k appears
        cost_matrix = np.zeros((num_episodes, num_episodes))

        for i in range(num_episodes):
            idx = 0
            for k in range(self.num_skills):
                cost = self.relabel_buffer["all_rewards"][i][:, k]
                cost_matrix[i, idx: idx + counts[k]] = np.sum(cost)
                idx += counts[k]

        # solve the linear sum assignment problem maximization, using scipy
        _, col_idx = optimize.linear_sum_assignment(cost_matrix, maximize=True)
        print(col_idx)

        # get new skill and new reward
        for i in range(num_episodes):
            # get one-hot encoding of skill
            skill = np.zeros(self.num_skills)
            skill[int(skill_array[col_idx[i]])] = 1
            max_rewards = self.relabel_buffer["all_rewards"][i][:, int(skill_array[col_idx[i]])][:, None]
            self.relabel_buffer['max_skill'][i] = skill
            self.relabel_buffer['max_rewards'][i] = max_rewards


    def _relabel_rl(self,
                    start_idx, end_idx, i_episode, new_skill):

        # print("Relabeling RL transitions")
        # relabel all transitions in episode
        new_rewards = self.relabel_buffer["max_rewards"][i_episode][None, :]
        # print(f"episode length = {end_idx + 1 - start_idx}, rewards length = {new_rewards.shape}")
        if start_idx > end_idx:
            # wrap around
            # replace skill and in all transitions and reward in last transition of episode
            self.locals["replay_buffer"].observations["skill"][start_idx:] = new_skill
            self.locals["replay_buffer"].observations["skill"][: end_idx + 1] = new_skill
            # change skill in next state
            self.locals["replay_buffer"].next_observations["skill"][start_idx:] = new_skill
            self.locals["replay_buffer"].next_observations["skill"][: end_idx + 1] = new_skill
            # change rewards
            # print(f'old: \n {self.locals["replay_buffer"].rewards[start_idx:]}\
            #                           {self.locals["replay_buffer"].rewards[: end_idx + 1]}')
            tmp_idx = self.locals["replay_buffer"].rewards[start_idx:].shape[0]
            self.locals["replay_buffer"].rewards[start_idx:] = new_rewards[:, : tmp_idx]
            self.locals["replay_buffer"].rewards[: end_idx + 1] = new_rewards[:, tmp_idx:]

            if self.prior_buffer:
                weight = np.sum(new_rewards)
                self.locals["replay_buffer"].weights[start_idx:] = weight
                self.locals["replay_buffer"].weights[: end_idx + 1] = weight

            # print(f'new: \n {self.locals["replay_buffer"].rewards[start_idx:]}\
            #                          {self.locals["replay_buffer"].rewards[: end_idx + 1]}')

        else:
            # replace skill and in all transitions and reward in last transition of episode
            self.locals["replay_buffer"].observations["skill"][start_idx: end_idx + 1] = new_skill
            # change skill in next state
            self.locals["replay_buffer"].next_observations["skill"][start_idx: end_idx + 1] = new_skill
            # change rewards
            # print(f'old: \n {self.locals["replay_buffer"].rewards[start_idx: end_idx + 1]}')
            self.locals["replay_buffer"].rewards[start_idx: end_idx + 1] = new_rewards

            if self.prior_buffer:
                print("relableing weights")
                weight = np.sum(new_rewards)
                print(f"weight = {weight}")
                self.locals["replay_buffer"].weights[start_idx: end_idx + 1] = weight

            # print(f'new: \n {self.locals["replay_buffer"].rewards[start_idx: end_idx + 1]}')


