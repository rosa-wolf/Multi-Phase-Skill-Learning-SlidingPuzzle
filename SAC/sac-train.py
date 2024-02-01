import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
import numpy as np

from stable_baselines3.common import noise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


import os
import sys
dir = os.path.dirname(__file__)
mod_dir = os.path.join(dir, "../gymframework/")
sys.path.append(mod_dir)
mod_dir = os.path.join(dir, "../")
sys.path.append(mod_dir)

from puzzle_env_skill_conditioned import PuzzleEnv

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# args for env
parser.add_argument('--env_name', type=str, default="skill_conditioned_2x2",
                    help='custom gym environment')
parser.add_argument('--snap_ratio', default=4., type=float,
                    help='1/Ratio of when symbolic state changes, if box is pushed')
parser.add_argument('--num_skills', default=8, type=int,
                    help='Number of skills policy should learn')
parser.add_argument('--num_steps', default=100, type=int,
                    help='Number of max episode steps')
parser.add_argument('--neg_dist_reward', action='store_true', default=False,
                    help='Give negative distance reward')
parser.add_argument('--movement_reward', action='store_true', default=False,
                    help='Give negative distance reward')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='Only sparse reward')
parser.add_argument('--reward_on_change', action='store_true', default=False,
                    help='Whether to give additional reward when box is pushed')
parser.add_argument('--term_on_change', action='store_true', default=False,
                    help='Terminate on change of symbolic state')
parser.add_argument('--random_init_board', action='store_true', default=False,
                    help='If true, it is not ensured that the skill execution is possible in the initial board configuration')
parser.add_argument('--reward_on_end', action='store_true', default=False,
                    help='Always give a reward on the terminating episode')
# args for SAC
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
#parser.add_argument('--eval', type=bool, default=True,
#                   help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='update coefficient for polyak update (default: 0.1)')
parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', # default=256
                    help='batch size (default: 256)')
parser.add_argument('--num_epochs', type=int, default=200000, metavar='N',
                    help='number of training epochs (default: 200000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_episode', type=int, default=50, metavar='N',
                    help='model updates per episode (default: 50)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Environment
if args.env_name.__contains__("1x2"):
    target_entropy = -2.5
    puzzle_path = '../Puzzles/slidingPuzzle_1x2.g'
    puzzle_size = [1, 2]
    skills = [np.array([[0, 1], [1, 0]])]

elif args.env_name.__contains__("2x2"):
    target_entropy = -3.
    puzzle_path = '../Puzzles/slidingPuzzle_2x2.g'
    puzzle_size = [2, 2]
    skills = [np.array([[0, 1], [1, 0], [2, 3], [3, 2]]),
              np.array([[0, 2], [2, 0], [1, 3], [3, 1]])]

elif args.env_name.__contains__("2x3"):
    target_entropy = -3.5
    puzzle_path = '../Puzzles/slidingPuzzle_2x3.g'
    puzzle_size = [2, 3]
    skills = [np.array([[0, 1], [1, 2], [3, 4], [4, 5]]),
              np.array([[1, 0], [2, 1], [5, 4], [4, 3]]),
              np.array([[0, 3], [3, 0], [1, 4], [4, 1], [2, 5], [5, 2]])]

elif args.env_name.__contains__("3x3"):
    target_entropy = -4.
    puzzle_path = '../Puzzles/slidingPuzzle_3x3.g'
    puzzle_size = [3, 3]
    skills = [np.array([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8]]),
              np.array([[1, 0], [2, 1], [4, 3], [5, 4], [7, 6], [8, 7]]),
              np.array([[0, 3], [3, 6], [1, 4], [4, 7], [2, 5], [5, 8]]),
              np.array([[3, 0], [6, 3], [4, 1], [7, 4], [5, 2], [8, 5]])]
else:
    raise ValueError("You must specify the environment to use")

env = PuzzleEnv(path=puzzle_path,
                max_steps=100,
                verbose=0,
                skills=skills,
                puzzle_size=puzzle_size,
                sparse_reward=args.sparse,
                reward_on_change=args.reward_on_change,
                neg_dist_reward=args.neg_dist_reward,
                movement_reward=args.movement_reward,
                term_on_change=True,
                reward_on_end=False,
                seed=args.seed)
eval_env = PuzzleEnv(path=puzzle_path,
                     skills=skills,
                     puzzle_size=puzzle_size,
                     max_steps=100,
                     verbose=0,
                     sparse_reward=args.sparse,
                     reward_on_change=args.reward_on_change,
                     neg_dist_reward=args.neg_dist_reward,
                     movement_reward=args.movement_reward,
                     term_on_change=False,
                     reward_on_end=False,
                     seed=98765)

eval_env = Monitor(eval_env)

env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

log_dir = "checkpoints/" + args.env_name + "_neg_dist" + str(args.neg_dist_reward) + "_movement" + str(args.movement_reward) + "_reward_on_change" + str(args.reward_on_change) + "_sparse" + str(args.sparse) + "_seed" + str(args.seed)
os.makedirs(log_dir, exist_ok=True)

env.reset()

# initialize callbacks
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path=log_dir + "/model/",
  name_prefix="model",
  save_replay_buffer=False,
  save_vecnormalize=True,
)

# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env,
                             log_path=log_dir, eval_freq=5000,
                             n_eval_episodes=10,
                             deterministic=True, render=False)

callbacks = CallbackList([checkpoint_callback, eval_callback])

# initialize SAC
model = SAC("MultiInputPolicy",  # could also use CnnPolicy
            env,        # gym env
            learning_rate=args.lr,  # same learning rate is used for all networks (can be fct of remaining progress)
            buffer_size=args.replay_size,
            learning_starts=1000, # when learning should start to prevent learning on little data
            batch_size=args.batch_size,  # mini-batch size for each gradient update
            #tau=args.tau,  # update for polyak update
            gamma=args.gamma, # discount factor
            gradient_steps=-1, # do as many gradient steps as steps done in the env
            #action_noise=noise.OrnsteinUhlenbeckActionNoise(),
            ent_coef='auto',
            target_entropy=target_entropy,
            #use_sde=True, # use state dependent exploration
            #use_sde_at_warmup=True, # use gSDE instead of uniform sampling at warmup
            stats_window_size=1,
            tensorboard_log=log_dir,
            device=device,
            verbose=1)

model.learn(total_timesteps=args.num_epochs * 100,
            log_interval=1,
            tb_log_name="tb_logs",
            progress_bar=False,
            callback=callbacks)

del model
env.close()
