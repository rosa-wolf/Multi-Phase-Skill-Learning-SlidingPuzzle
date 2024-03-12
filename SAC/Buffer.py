import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    ReplayBufferSamples,
)

from BufferSamples import (
    PrioritizedReplayBufferSamples,
    PrioritizedDictReplayBufferSamples,
)

from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

class PriorityReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray
    weights: np.ndarray
    enumeration: np.ndarray
    last_done: np.int8

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        zeta: float = 0.5,
        recent: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.weights = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.enumeration = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.last_done = None

        # threshold value for SDP sampling
        self.zeta = zeta
        # number of samples that should be most recent ones
        self.num_recent = recent

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        #print(f"obs shape = {obs.shape}, done shape = {done.shape}, reward shape = {reward.shape}")

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.enumeration[self.pos] = np.array(self.pos)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
        if done[0]:
            # process episode transitions, by adding return as weight to each transition of episode
            self._add_weights(self.pos)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _calculate_episode_return(self, start_idx: np.int8, end_idx: np.int8):
        total_return = 0
        # if skill execution is not possible give highly negative weight, as this episode is not as interesting
        if np.all(self.rewards[start_idx: end_idx] == 0):
            total_return = -1000.
            return total_return

        if start_idx > end_idx:
            # episode wraps around buffer end
            total_return += np.sum(self.rewards[start_idx:])
            total_return += np.sum(self.rewards[: end_idx + 1])
        elif start_idx == end_idx:
            total_return += np.sum(self.rewards[end_idx])
        else:
            total_return += np.sum(self.rewards[start_idx: end_idx + 1])

        return total_return

    def _add_weights(self, end_idx: np.int8):
        if self.last_done is None:
            # we are processing first ever episode
            start_idx = 0
        else:
            if self.last_done == self.buffer_size - 1:
                # last episde exactly filled buffer
                start_idx = 0
            else:
                start_idx = self.last_done + 1

        episode_return = self._calculate_episode_return(start_idx, end_idx)
        #print(f"episode_return = {episode_return}")

        # add return as weight to all transitions
        if start_idx > end_idx:
            self.weights[start_idx:] = episode_return
            self.weights[: end_idx + 1] = episode_return
        elif start_idx == end_idx:
            self.weights[end_idx] = episode_return
        else:
            self.weights[start_idx: end_idx + 1] = episode_return

        self.last_done = end_idx



    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> PrioritizedReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        # sample 2 batches and take prioritized samples over those two batches
        batches = []
        for _ in range(2):
            if not self.optimize_memory_usage:
                batches.append(super().sample(batch_size=batch_size, env=env))
            # Do not sample the element with index `self.pos` as the transitions is invalid
            # (we use only one array to store `obs` and `next_obs`)
            else:
                if self.full:
                    batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
                else:
                    batch_inds = np.random.randint(0, self.pos, size=batch_size)
                batches.append(self._get_samples(batch_inds, env=env))

        buffer = self._get_prior_batch(batches[0], batches[1], batch_size)

        # replace x samples with most recent ones
        recent = self._get_recent_samples(buffer, batch_size, env=env)

        #print("Number of trans in buffer = ", recent.weights.shape)
        return recent

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.weights[batch_inds, env_indices],
            self.enumeration[batch_inds, env_indices],
        )
        return PrioritizedReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def _get_prior_batch(self, batch1, batch2, batch_size: int):
        # calculate score
        score = batch1.weights @ batch2.weights / (th.norm(batch1.weights) * th.norm(batch2.weights))
        if score > self.zeta and False:
            # take random of the two batches
            if np.random.uniform() >= 0.5:
                return batch1
            return batch2

        # delete transitions from batch 2 that are also in batch1
        idx = th.where(th.isin(batch2.enumeration, batch1.enumeration) == False)[0]
        if idx.shape != (0,):
            batch2 = PrioritizedReplayBufferSamples(batch2.observations[idx],
                                                    batch2.actions[idx],
                                                    batch2.next_observations[idx],
                                                    batch2.dones[idx],
                                                    batch2.rewards[idx],
                                                    batch2.weights[idx],
                                                    batch2.enumeration[idx])

        # get transitions with max weights
        b1_weights = batch1.weights
        b2_weights = batch2.weights

        tmp = th.concatenate((b1_weights, b2_weights))

        _, max_idx = th.topk(tmp, batch_size)
        # get which idx was originally from which of the two batches
        idx_batch1 = max_idx[th.where(max_idx < b1_weights.shape[0])]
        idx_batch2 = max_idx[th.where(max_idx >= b1_weights.shape[0])]
        idx_batch2 -= b1_weights.shape[0]

        #print(f"batch_1 =\n {batch1}\n idx_batch1 = {idx_batch1} \n batch_2 = \n {batch2}\nidx_batch2 = {idx_batch2}")

        # new bach contains elements form batch1 at position idx_batch1
        # and elements from batch2 (where repitions have been removed) at position idx_batch2
        observations = th.concatenate((batch1.observations[idx_batch1], batch2.observations[idx_batch2]))
        actions = th.concatenate((batch1.actions[idx_batch1], batch2.actions[idx_batch2]))
        next_observations = th.concatenate((batch1.next_observations[idx_batch1], batch2.next_observations[idx_batch2]))
        dones = th.concatenate((batch1.dones[idx_batch1], batch2.dones[idx_batch2]))
        rewards = th.concatenate((batch1.rewards[idx_batch1], batch2.rewards[idx_batch2]))
        weights = th.concatenate((batch1.weights[idx_batch1], batch2.weights[idx_batch2]))
        enumeration = th.concatenate((batch1.enumeration[idx_batch1], batch2.enumeration[idx_batch2]))

        return PrioritizedReplayBufferSamples(observations,
                                              actions,
                                              next_observations,
                                              dones,
                                              rewards,
                                              weights,
                                              enumeration)

    def _get_recent_samples(self, batch, batch_size, env: Optional[VecNormalize] = None) \
            -> PrioritizedReplayBufferSamples:
        if self.num_recent > batch_size:
            raise ValueError("The number of recent transitions to take in the batch is higher than the batch size.")

        # Sample transitions, that should not be replaced
        idx = np.random.uniform(0, batch_size, size=(batch_size - self.num_recent,))

        if self.pos < 1000:
          batch_inds = np.concatenate((np.arange(self.pos),
                                      np.arange(self.buffer_size - (1000 - self.pos), self.buffer_size)))
        else:
           batch_inds = np.arange(self.pos - 1000, self.pos)

        print(batch_inds)

        batch_inds = np.random.choice(batch_inds, self.num_recent)
        print(f"sampled = {batch_inds}")


        # sample num recent transitions from last 1000 transitions
        # takes always num_recent most recent samples
        #if self.pos < self.num_recent:
        #   batch_inds = np.concatenate((np.arange(self.pos),
        #                               np.arange(self.buffer_size - (self.num_recent - self.pos), self.buffer_size)))
        #else:
        #    batch_inds = np.arange(self.pos - self.num_recent, self.pos)

        recent_batch = self._get_samples(batch_inds, env=env)

        observations = th.concatenate((batch.observations[idx], recent_batch.observations))
        actions = th.concatenate((batch.actions[idx], recent_batch.actions))
        next_observations = th.concatenate((batch.next_observations[idx], recent_batch.next_observations))
        dones = th.concatenate((batch.dones[idx], recent_batch.dones))
        rewards = th.concatenate((batch.rewards[idx], recent_batch.rewards))
        weights = th.concatenate((batch.weights[idx], recent_batch.weights))
        enumeration = th.concatenate((batch.enumeration[idx], recent_batch.enumeration))

        return PrioritizedReplayBufferSamples(observations,
                                              actions,
                                              next_observations,
                                              dones,
                                              rewards,
                                              weights,
                                              enumeration)

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype

class PriorityDictReplayBuffer(PriorityReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observation_space: spaces.Dict
    obs_shape: Dict[str, Tuple[int, ...]]  # type: ignore[assignment]
    observations: Dict[str, np.ndarray]  # type: ignore[assignment]
    next_observations: Dict[str, np.ndarray]  # type: ignore[assignment]

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            zeta: float = 0.5,
            recent: int = 10,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super(PriorityReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert not optimize_memory_usage, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.weights = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.enumeration = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.last_done = None

        # threshold value for SDP sampling
        self.zeta = zeta
        # number of samples that should be most recent ones
        self.num_recent = recent

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage: float = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if not optimize_memory_usage:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(  # type: ignore[override]
            self,
            obs: Dict[str, np.ndarray],
            next_obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key])

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.enumeration[self.pos] = np.array(self.pos)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
        if done[0]:
            # process episode transitions, by adding return as weight to each transition of episode
            self._add_weights(self.pos)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(  # type: ignore[override]
            self,
            batch_size: int,
            env: Optional[VecNormalize] = None,
    ) -> PrioritizedDictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super().sample(batch_size=batch_size, env=env)

    def _get_samples(  # type: ignore[override]
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()},
                                   env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return PrioritizedDictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(
                self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
            weights=self.to_torch(self.weights[batch_inds, env_indices]),
            enumeration=self.to_torch(self.enumeration[batch_inds, env_indices]),
        )

    def _get_prior_batch(self, batch1, batch2, batch_size: int) -> PrioritizedDictReplayBufferSamples:
        # calculate score
        #weight = th.ones((batch_size))
        #norm = 1
        #for batch in batches:
        #    weight *= batch.weights
        #    norm *= th.norm(batch.weights)

        #score = th.sum(weight) / norm

        score = batch1.weights @ batch2.weights / (th.norm(batch1.weights) * th.norm(batch2.weights))
        #print(f"score = {score}")
        if score > self.zeta:
            #print("taking one batch")
            # take random of the two batches
            if np.random.uniform() >= 0.5:
                return batch1
            return batch2

        # delete transitions from batch 2 that are also in batch1
        idx = th.where(th.isin(batch2.enumeration, batch1.enumeration) == False)[0]
        if idx.shape != (0,):
            observations = {key: batch2.observations[key][idx] for key in batch2.observations.keys()}
            next_observations = {key: batch2.next_observations[key][idx] for key in batch2.next_observations.keys()}
            batch2 = PrioritizedDictReplayBufferSamples(observations,
                                                        batch2.actions[idx],
                                                        next_observations,
                                                        batch2.dones[idx],
                                                        batch2.rewards[idx],
                                                        batch2.weights[idx],
                                                        batch2.enumeration[idx])

        # get transitions with max weights
        b1_weights = batch1.weights
        b2_weights = batch2.weights

        tmp = th.concatenate((b1_weights, b2_weights))

        _, max_idx = th.topk(tmp, batch_size)
        # get which idx was originally from which of the two batches
        idx_batch1 = max_idx[th.where(max_idx < b1_weights.shape[0])]
        idx_batch2 = max_idx[th.where(max_idx >= b1_weights.shape[0])]
        idx_batch2 -= b1_weights.shape[0]

        # print(f"batch_1 =\n {batch1}\n idx_batch1 = {idx_batch1} \n batch_2 = \n {batch2}\nidx_batch2 = {idx_batch2}")

        # new bach contains elements form batch1 at position idx_batch1
        # and elements from batch2 (where repitions have been removed) at position idx_batch2
        observations = {
            key: th.concatenate((batch1.observations[key][idx_batch1], batch2.observations[key][idx_batch2]))
            for key in batch1.observations.keys()}
        actions = th.concatenate((batch1.actions[idx_batch1], batch2.actions[idx_batch2]))
        next_observations = {
            key: th.concatenate((batch1.next_observations[key][idx_batch1], batch2.next_observations[key][idx_batch2]))
            for key in batch1.next_observations.keys()}
        dones = th.concatenate((batch1.dones[idx_batch1], batch2.dones[idx_batch2]))
        rewards = th.concatenate((batch1.rewards[idx_batch1], batch2.rewards[idx_batch2]))
        weights = th.concatenate((batch1.weights[idx_batch1], batch2.weights[idx_batch2]))
        enumeration = th.concatenate((batch1.enumeration[idx_batch1], batch2.enumeration[idx_batch2]))

        return PrioritizedDictReplayBufferSamples(observations,
                                                  actions,
                                                  next_observations,
                                                  dones,
                                                  rewards,
                                                  weights,
                                                  enumeration)

    def _get_recent_samples(self, batch, batch_size, env: Optional[VecNormalize] = None) \
            -> PrioritizedDictReplayBufferSamples:
        if self.num_recent > batch_size:
            raise ValueError("The number of recent transitions to take in the batch is higher than the batch size.")

        # Sample transitions, that should not be replaced
        idx = np.random.uniform(0, batch_size, size=(batch_size - self.num_recent,))

        # self.pos is in front of last added transition
        if self.pos < 100:
            batch_inds = np.concatenate((np.arange(self.pos),
                                         np.arange(self.buffer_size - (100 - self.pos), self.buffer_size)))
        else:
            batch_inds = np.arange(self.pos - 100, self.pos)

        print(batch_inds)

        batch_inds = np.random.choice(batch_inds, self.num_recent)
        recent_batch = self._get_samples(batch_inds, env=env)

        print(f"sampled = {batch_inds}")

        observations = {
            key: th.concatenate((batch.observations[key][idx], recent_batch.observations[key]))
            for key in batch.observations.keys()}
        actions = th.concatenate((batch.actions[idx], recent_batch.actions))
        next_observations = {
            key: th.concatenate((batch.next_observations[key][idx], recent_batch.next_observations[key]))
            for key in batch.next_observations.keys()}
        dones = th.concatenate((batch.dones[idx], recent_batch.dones))
        rewards = th.concatenate((batch.rewards[idx], recent_batch.rewards))
        weights = th.concatenate((batch.weights[idx], recent_batch.weights))
        enumeration = th.concatenate((batch.enumeration[idx], recent_batch.enumeration))

        return PrioritizedDictReplayBufferSamples(observations,
                                                  actions,
                                                  next_observations,
                                                  dones,
                                                  rewards,
                                                  weights,
                                                  enumeration)
class SeadsBuffer(DictReplayBuffer):
    observation_space: spaces.Dict
    obs_shape: Dict[str, Tuple[int, ...]]  # type: ignore[assignment]
    observations: Dict[str, np.ndarray]  # type: ignore[assignment]
    next_observations: Dict[str, np.ndarray]  # type: ignore[assignment]

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):

        super().__init__(buffer_size,
                         observation_space,
                         action_space,
                         device,
                         n_envs,
                         optimize_memory_usage,
                         handle_timeout_termination)

        self.recent_samples = 25600

    def sample(  # type: ignore[override]
            self,
            batch_size: int,
            env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.full:
            num_samples = self.recent_samples if self.pos > 256 else self.pos
        else:
            num_samples = self.recent_samples

        print(f"num_samples = {num_samples}")

        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            long_batch_inds = np.random.randint(0, upper_bound, size=num_samples)
        else:
            if self.full:
                long_batch_inds = (np.random.randint(1, self.buffer_size, size=num_samples) + self.pos) % self.buffer_size
            else:
                long_batch_inds = np.random.randint(0, self.pos, size=num_samples)


        # get recent samples
        recent_inds = self._get_recent_inds(num_samples)

        batch_inds = np.concatenate((long_batch_inds, recent_inds))

        return self._get_samples(batch_inds, env)

    def _get_recent_inds(self,
                         num_samples):
        """
        get all most recent samples from buffer
        """

        # self.pos is in front of last added transition
        if self.pos < num_samples:
            inds = np.concatenate((np.arange(self.pos),
                                         np.arange(self.buffer_size - (num_samples - self.pos), self.buffer_size)))
        else:
            inds = np.arange(self.pos - num_samples, self.pos)

        return inds




