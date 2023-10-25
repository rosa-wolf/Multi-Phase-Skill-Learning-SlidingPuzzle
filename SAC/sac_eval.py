import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import argparse

import torch
import numpy as np

import os
import sys
dir = os.path.dirname(__file__)
mod_dir = os.path.join(dir, "../gymframework/")
sys.path.append(mod_dir)
mod_dir = os.path.join(dir, "../")
sys.path.append(mod_dir)

from puzzle_env_stable_baselines import PuzzleEnv

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# args for env
parser.add_argument('--env_name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--skill', default=0, type=int,
                    help='Enumeration of the skill to train')
parser.add_argument('--vel_steps', default=1, type=int,
                    help='Number of times to apply velocity control in one step of the agent')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='Only sparse reward')
parser.add_argument('--give_sym_obs', action='store_true', default=False,
                    help='Penalize pushing without change of symbolic observation')
parser.add_argument('--z_cov', type=float, default=10, metavar='G',
                    help='Parameter for inverse cov of optimal z-position function (default: 10)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_steps', type=int, default=100, metavar='N',
                    help='maximum number of steps (default: 100)')

# args for SAC
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
#parser.add_argument('--eval', type=bool, default=True,
#                   help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', # default=256
                    help='batch size (default: 256)')
parser.add_argument('--num_epochs', type=int, default=200000, metavar='N',
                    help='number of training epochs (default: 200000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
env = PuzzleEnv(path='../slidingPuzzle.g', skill=0, max_steps=args.num_steps, random_init_pos=True,
                random_init_config=True, verbose=1, give_sym_obs=False, sparse_reward=args.sparse, z_cov=args.z_cov)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.cuda:
    device = 'cuda'
else:
    device = 'cpu'

checkpoint_name = args.env_name + "_" + str(args.num_epochs) + "epochs_sparse" + str(args.sparse) + "_seed" + str(
        args.seed) + "_vel_steps" + str(args.vel_steps)

# load SAC model
path = "skill0_10000epochs_sparseFalse_seed123456_vel_steps1.zip"
model = SAC.load(path)

obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    next_state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

del model