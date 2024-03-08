from stable_baselines3 import SAC
import argparse
import torch
import numpy as np
import logging as lg
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from callbacks.fm_callback import FmCallback
from callbacks.my_eval_callback import EvalCallback
from gymframework.puzzle_env_skill_conditioned_parallel_training import PuzzleEnv
from forwardmodel_simple_input.forward_model import ForwardModel

import os

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
parser.add_argument('--prior_buffer', action='store_true', default=False,
                    help='Whether to use priority buffer')
parser.add_argument('--second_best', action='store_true', default=False,
                    help='Whether to do second best normalization in calculation of reward')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='Only sparse reward')
parser.add_argument('--doinit', action='store_true', default=False,
                    help='Whether to do initial phase')
parser.add_argument('--dorefinement', action='store_true', default=False,
                    help='Whether to do refinement phase')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_steps', type=int, default=100, metavar='N',
                    help='maximum number of steps (default: 100)')
parser.add_argument('--reward_on_change', action='store_true', default=False,
                    help='Whether to give additional reward when boxes is pushed')
parser.add_argument('--term_on_change', action='store_true', default=False,
                    help='Terminate on change of symbolic state')
parser.add_argument('--random_init_board', action='store_true', default=False,
                    help='If true, it is not ensured that the skill execution is possible in the initial board configuration')
parser.add_argument('--reward_on_end', action='store_true', default=False,
                    help='Always give a reward on the terminating episode')
parser.add_argument('--snap_ratio', default=4., type=int,
                    help='1/Ratio of when symbolic state changes, if boxes is pushed')

# args for SAC
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
#parser.add_argument('--eval', type=bool, default=True,
#                   help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.1, metavar='G',
                    help='update coefficient for polyak update (default: 0.1)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
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

if args.prior_buffer:
    from Buffer import PriorityDictReplayBuffer
    buffer_class = PriorityDictReplayBuffer
else:
    from Buffer import SeadsBuffer
    buffer_class = SeadsBuffer

log_dir = "checkpoints/" + "parallel" + args.env_name + "_num_skills" + str(args.num_skills) + "_sparse" + str(args.sparse) + "_relabeling" + str(args.relabeling) + "_priorbuffer" + str(args.prior_buffer) + "_seed" + str(args.seed)
os.makedirs(log_dir, exist_ok=True)
fm_dir = log_dir + "/fm"
os.makedirs(fm_dir, exist_ok=True)

max_steps = 100
# Environment
if args.env_name.__contains__("1x2"):
    target_entropy = -2.5
    puzzle_path = '../Puzzles/slidingPuzzle_1x2.g'
    puzzle_size = [1, 2]

elif args.env_name.__contains__("2x2"):
    target_entropy = -3.
    puzzle_path = '../Puzzles/slidingPuzzle_2x2.g'
    puzzle_size = [2, 2]

elif args.env_name.__contains__("2x3"):
    target_entropy = -3.5
    puzzle_path = '../Puzzles/slidingPuzzle_2x3.g'
    puzzle_size = [2, 3]

elif args.env_name.__contains__("3x3"):
    target_entropy = -4.5
    puzzle_path = '../Puzzles/slidingPuzzle_3x3.g'
    puzzle_size = [3, 3]
    max_steps = 200

else:
    raise ValueError("You must specify the environment to use")

# initialize save for fm
# Create folder if needed
# initialize dummy fm
os.makedirs(fm_dir, exist_ok=True)
# save fm
fm = ForwardModel(num_skills=args.num_skills, puzzle_size=puzzle_size)
torch.save(fm.model.state_dict(), fm_dir + "/fm")

env = PuzzleEnv(path=puzzle_path,
                max_steps=max_steps,
                second_best=args.second_best,
                num_skills=args.num_skills,
                init_phase=args.doinit,
                refinement_phase=args.dorefinement,
                verbose=0,
                fm_path=fm_dir + "/fm",
                puzzlesize=puzzle_size,
                sparse_reward=args.sparse,
                reward_on_change=True,
                term_on_change=True,
                reward_on_end=True,
                relabel=args.relabeling,
                seed=args.seed,
                snapRatio=args.snap_ratio)

eval_env = PuzzleEnv(path=puzzle_path,
                max_steps=max_steps,
                num_skills=args.num_skills,
                second_best=args.second_best,
                verbose=0,
                fm_path=fm_dir + "/fm",
                puzzlesize=puzzle_size,
                sparse_reward=args.sparse,
                reward_on_change=True,
                term_on_change=True,
                reward_on_end=True,
                relabel=args.relabeling,
                seed=987654,
                snapRatio=args.snap_ratio)
eval_env = Monitor(eval_env)

env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

lg.basicConfig(filename=log_dir + "/change.log", level=lg.INFO, filemode='w',
                                format='%(name)s - %(levelname)s - %(message)s')

# initialize callbacks
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=10000,
  save_path=log_dir + "/model/",
  name_prefix="model",
  save_replay_buffer=False,
  save_vecnormalize=True,
)

# callback for updating and training fm
fm_callback = FmCallback(update_freq=1000,
                         env=env,
                         save_path=log_dir + "/fm",
                         size=puzzle_size,
                         num_skills=args.num_skills,
                         prior_buffer=args.prior_buffer,
                         seed=args.seed,
                         relabel=args.relabeling)

# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env,
                             log_path=log_dir, eval_freq=5000,
                             n_eval_episodes=0,
                             deterministic=True, render=False)

callback = CallbackList([checkpoint_callback, fm_callback, eval_callback])

# initialize SAC
model = SAC('MultiInputPolicy',
            env,        # gym env
            learning_rate=args.lr,  # same learning rate is used for all networks (can be fct of remaining progress)
            buffer_size=2048 * args.num_steps,
            replay_buffer_class=buffer_class,
            learning_starts=256 * args.num_steps, # when learning should start to prevent learning on little data
            batch_size=args.batch_size,  # mini-batch size for each gradient update
            #tau=args.tau,  # update for polyak update
            gamma=args.gamma,  # learning rate
            gradient_steps=-1, # do as many gradient steps as steps done in the env
            train_freq=(args.num_episodes, "episode"),
            #action_noise=noise.OrnsteinUhlenbeckActionNoise(),
            ent_coef='auto',
            target_entropy=target_entropy,
            #use_sde=True, # use state dependent exploration
            #use_sde_at_warmup=True, # use gSDE instead of uniform sampling at warmup
            #stats_window_size=args.batch_size,
            tensorboard_log=log_dir,
            #policy_kwargs={'net_arch': [256, 256, 256]},
            device=device,
            verbose=1)

model.learn(total_timesteps=args.num_epochs * 100,
            log_interval=10,
            tb_log_name="tb_logs",
            progress_bar=True,
            callback=callback)

#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
#print(f"mean_reward = {mean_reward}, std_reward = {std_reward}")

del model
env.close()
