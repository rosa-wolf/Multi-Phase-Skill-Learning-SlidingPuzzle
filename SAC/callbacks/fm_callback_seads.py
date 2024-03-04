import os

import gymnasium as gym
import numpy as np
import torch
import logging as lg
import time

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
                 sample_size=32,
                 size=[1, 2],
                 num_skills=2,
                 relabel=False,
                 train_fm=True,
                 logging=False,
                 eval_freq=500,
                 verbose=0):

        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_path = save_path
        self.sample_size = sample_size
        self.relabel = relabel
        self.relabel_buffer = {"max_rewards": [],
                                "max_skill": [],
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
        self.fm = ForwardModel(num_skills=self.num_skills, puzzle_size=size, batch_size=sample_size, learning_rate=0.001)

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
        num_updates = 4

        # make new buffer from recent and sampled long-term episodes
        # only update fm once 256 episodes have been sampled
        if len(self.long_term_buffer) >= 256:
            final_fm_buffer = FmReplayMemory(512, self.seed)
            recent_state_batch, recent_skill_batch, recent_next_state_batch = self.recent_buffer.sample(256)
            state_batch, skill_batch, next_state_batch = self.long_term_buffer.sample(256)

            for i in range(256):
                final_fm_buffer.push(state_batch[i], skill_batch[i], next_state_batch[i])
                final_fm_buffer.push(recent_state_batch[i], recent_skill_batch[i], recent_next_state_batch[i])

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
            np.savez(self.save_path + "log_data", test_acc=test_acc, train_acc=train_acc)

            if self.verbose > 0:
                print(f'\tEval Loss: {test_loss:.3f} | Eval Acc: {test_acc * 100:.2f}%')


    def _on_step(self):

        if self.locals["dones"] != 0:
            self.relabel_buffer["max_rewards"].append((self.locals["infos"][0])["max_rewards"])
            self.relabel_buffer["max_skill"].append((self.locals["infos"][0])["max_skill"])
            self.relabel_buffer["episode_length"].append(((self.locals["infos"][0])["episode"]["l"]))
            self.relabel_buffer["total_num_steps"].append(self.n_calls)

        # only start training after buffer is filled a bit
        if self.train_fm:
            if self.n_calls % self.update_freq == 0:
                self._train_fm()

            if self.logging:
                if self.n_calls % self.eval_freq == 0:
                    self._eval_fm()

    def _on_rollout_end(self) -> bool:
        """
        Update fm
        This function is called before the sac networks are updated
        """

        num_relabel = len(self.relabel_buffer["total_num_steps"])

        print(f"==================\nnum_relabel = {num_relabel}")

        # for now relabel without caring for skill distribution
        # dones = np.where((self.locals["replay_buffer"]).dones == 1)[0]

        for i_episode in range(num_relabel):
            ## get start and end idx of episode to relabel
            # if dones.shape[0] < (num_relabel + 1) - i_episode:
            #    start_idx = 0
            # else:
            #    start_idx = dones[-(num_relabel + 1) + i_episode] + 1

            # end_idx = dones[-(num_relabel) + i_episode]

            # print(f"steps = {self.relabel_buffer['total_num_steps']}, epi length = {self.relabel_buffer['episode_length']}")
            print(f"i_episode = {i_episode}")
            print(f"total_num_steps = {self.relabel_buffer['total_num_steps']}")
            start_idx = self.relabel_buffer["total_num_steps"][i_episode] - self.relabel_buffer["episode_length"][
                i_episode]
            end_idx = self.relabel_buffer["total_num_steps"][i_episode] - 1

            # print(f"before wrapping: start_idx = {start_idx}, end_idx = {end_idx}")
            start_idx = start_idx % self.locals["replay_buffer"].buffer_size
            end_idx = end_idx % self.locals["replay_buffer"].buffer_size

            # wrap around end of replay buffer
            # print(f"start_idx = {start_idx}, end_idx = {end_idx}")

            init_empty = (self.locals["replay_buffer"]).next_observations["init_empty"][end_idx]
            out_empty = (self.locals["replay_buffer"]).next_observations["curr_empty"][end_idx]
            old_skill = (self.locals["replay_buffer"]).next_observations["skill"][end_idx]
            # print(f"init_empty = {init_empty}, out_empty = {out_empty}")

            if self.relabel:
                # get skill that maximizes reward
                # TODO: relabeling for now assumes that we terminated on change of symbolic state
                new_skill = self.relabel_buffer["max_skill"][i_episode]

                # print(f"old_skill = {old_skill}, new_skill = {new_skill}")
                # print(f"start_idx = {start_idx}, end_idx = {end_idx}")
                # never relabel policy transitions
                if not (new_skill == old_skill).all():
                    # relabel policy transitions with 50% probability
                    if np.random.normal() > 0.5:
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

                            # print(f'new: \n {self.locals["replay_buffer"].rewards[start_idx: end_idx + 1]}')

                # always relabel fm transition
                if self.train_fm:
                    self.long_term_buffer.push(init_empty.flatten(), new_skill.flatten(), out_empty.flatten())
                    self.recent_buffer.push(init_empty.flatten(), new_skill.flatten(), out_empty.flatten())
                    # print(f"init = {init_empty.flatten()}, new_skill = {new_skill.flatten()}, old_skill = {old_skill.flatten()}, out_empty = {out_empty.flatten()}")

            else:
                if self.train_fm:
                    self.long_term_buffer.push(init_empty.flatten(), old_skill.flatten(), out_empty.flatten())
                    self.recent_buffer.push(init_empty.flatten(), old_skill.flatten(), out_empty.flatten())

        # print("-------------------------------------------------------")

        self.relabel_buffer = {"max_rewards": [],
                               "max_skill": [],
                               "episode_length": [],
                               "total_num_steps": []}


