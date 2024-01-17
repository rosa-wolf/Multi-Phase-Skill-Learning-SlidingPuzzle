import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
import numpy as np

from stable_baselines3.common import noise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, StopTrainingOnMaxEpisodes

from callbacks.fm_callback import FmCallback

import os
import sys
dir = os.path.dirname(__file__)
mod_dir = os.path.join(dir, "../gymframework/")
sys.path.append(mod_dir)
mod_dir = os.path.join(dir, "../")
sys.path.append(mod_dir)

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# args for env
parser.add_argument('--env_name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--num_skills', default=2, type=int,
                    help='Enumeration of the skill to train')
parser.add_argument('--num_episodes', default=1, type=int,
                    help='Number of episode to collect in each rollout')
parser.add_argument('--vel_steps', default=1, type=int,
                    help='Number of times to apply velocity control in one step of the agent')
parser.add_argument('--relabeling', action='store_true', default=False,
                    help='Do HER for higher level skills')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='Only sparse reward')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_steps', type=int, default=100, metavar='N',
                    help='maximum number of steps (default: 100)')
parser.add_argument('--reward_on_change', action='store_true', default=False,
                    help='Whether to give additional reward when box is pushed')
parser.add_argument('--term_on_change', action='store_true', default=False,
                    help='Terminate on change of symbolic state')
parser.add_argument('--random_init_board', action='store_true', default=False,
                    help='If true, it is not ensured that the skill execution is possible in the initial board configuration')
parser.add_argument('--reward_on_end', action='store_true', default=False,
                    help='Always give a reward on the terminating episode')
parser.add_argument('--snap_ratio', default=4., type=int,
                    help='1/Ratio of when symbolic state changes, if box is pushed')

# args for SAC
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
#parser.add_argument('--eval', type=bool, default=True,
#                   help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.1, metavar='G',
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

log_dir = "checkpoints/" + args.env_name + "_num_skills" + str(args.num_skills) + "_relabeling" + str(args.relabeling)
os.makedirs(log_dir, exist_ok=True)
fm_dir = log_dir + "/fm"
os.makedirs(fm_dir, exist_ok=True)


# TODO: make this a parameter
relabel = True


match args.env_name:
    case "parallel_1x2":
        from puzzle_env_small_skill_conditioned_parallel_training import PuzzleEnv
        env = PuzzleEnv(path='../Puzzles/slidingPuzzle_1x2.g',
                        max_steps=100,
                        num_skills=args.num_skills,
                        verbose=0,
                        fm_path=fm_dir + "/fm",
                        sparse_reward=True,
                        reward_on_change=True,
                        term_on_change=True,
                        reward_on_end=True,
                        relabel=args.relabeling,
                        snapRatio=args.snap_ratio)
        puzzle_size = [1, 2]
    case "parallel_2x2":
        from puzzle_env_2x2_skill_conditioned_parallel_training import PuzzleEnv
        env = PuzzleEnv(path='../Puzzles/slidingPuzzle_2x2.g',
                        max_steps=100,
                        num_skills=args.num_skills,
                        verbose=0,
                        fm_path=fm_dir + "/fm",
                        sparse_reward=True,
                        reward_on_change=False,
                        term_on_change=False,
                        reward_on_end=args.reward_on_end,
                        snapRatio=args.snap_ratio)

        puzzle_size = [2, 2]

check_env(env)

env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

checkpoint_name = args.env_name + "_" + "_sparse" + str(args.sparse) + "_seed" + str(
        args.seed) + "_num_skills" + str(args.num_skills)

# initialize callbacks
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path=log_dir + "/model/",
  name_prefix="model",
  save_replay_buffer=False,
  save_vecnormalize=True,
)

# callback for updating and training fm
fm_callback = FmCallback(update_freq=500,
                         save_path=log_dir + "/fm",
                         size=puzzle_size,
                         num_skills=args.num_skills,
                         seed=args.seed,
                         relabel=relabel)

callback = CallbackList([checkpoint_callback, fm_callback])

# initialize SAC
model = SAC("MultiInputPolicy",
            env,        # gym env
            learning_rate=args.lr,  # same learning rate is used for all networks (can be fct of remaining progress)
            buffer_size=args.replay_size,
            learning_starts=1000, # when learning should start to prevent learning on little data
            batch_size=args.batch_size,  # mini-batch size for each gradient update
            #tau=args.tau,  # update for polyak update
            gamma=args.gamma,  # learning rate
            gradient_steps=-1, # do as many gradient steps as steps done in the env
            train_freq=(args.num_episodes, "episode"),
            #action_noise=noise.OrnsteinUhlenbeckActionNoise(),
            ent_coef='auto',
            #use_sde=True, # use state dependent exploration
            #use_sde_at_warmup=True, # use gSDE instead of uniform sampling at warmup
            #stats_window_size=args.batch_size,
            tensorboard_log=log_dir,
            device=device,
            verbose=1)

model.learn(total_timesteps=args.num_epochs * 100,
            log_interval=10,
            tb_log_name="tb_logs",
            progress_bar=True,
            callback=callback)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"mean_reward = {mean_reward}, std_reward = {std_reward}")

del model
env.close()
