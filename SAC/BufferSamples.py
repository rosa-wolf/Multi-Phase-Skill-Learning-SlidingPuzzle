"""Common aliases for type hints"""
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Protocol, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th

# Avoid circular imports, we use type hint as string to avoid it too
if TYPE_CHECKING:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv

GymEnv = Union[gym.Env, "VecEnv"]
GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymResetReturn = Tuple[GymObs, Dict]
AtariResetReturn = Tuple[np.ndarray, Dict[str, Any]]
GymStepReturn = Tuple[GymObs, float, bool, bool, Dict]
AtariStepReturn = Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]
TensorDict = Dict[str, th.Tensor]
OptimizerStateDict = Dict[str, Any]
MaybeCallback = Union[None, Callable, List["BaseCallback"], "BaseCallback"]
PyTorchObs = Union[th.Tensor, TensorDict]

# A schedule takes the remaining progress as input
# and ouputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]

class PrioritizedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    weights: th.Tensor
    enumeration: th.Tensor