import numpy as np
import torch as th
import argparse
import time

from stable_baselines3 import SAC

from forwardmodel_simple_input.forward_model import ForwardModel
from gymframework.puzzle_env_skill_conditioned_parallel_training import PuzzleEnv

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--num_skills', default=8, type=int,
                    help='Number of skills policy should learn')
args = parser.parse_args()

if th.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# for making a video
# first ensure a folder
import os
os.system('mkdir -p z.vid')

# goal state for 2x2 puzzle cannot be random, because then for most initial configs the goal would not be reachable
init_state = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
goal_state = np.array([[0, 0, 0, 1],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0]])

# load forward model and policy
fm = ForwardModel(width=2,
                  height=2,
                  num_skills=args.num_skills,
                  batch_size=10,
                  learning_rate=0.001)
fm_path = "/home/rosa/Documents/Uni/Masterarbeit/parallel2x2-new-penalty_num_skills2_sparseTrue_relabelingFalse/fm/fm"
fm.model.load_state_dict(th.load(fm_path, weights_only=True))
fm.model.eval()

print(fm.get_full_pred())

env = PuzzleEnv(path='Puzzles/slidingPuzzle_2x2.g',
                        max_steps=100,
                        num_skills=args.num_skills,
                        logging=False,
                        verbose=1,
                        fm_path=fm_path,
                        train_fm=False,
                        sparse_reward=True,
                        reward_on_change=True,
                        term_on_change=True,
                        reward_on_end=False)

model = SAC.load("/home/rosa/Documents/Uni/Masterarbeit/parallel2x2-new-penalty_num_skills2_sparseTrue_relabelingFalse/model/model_330000_steps", env=env)



# plan skills to go from init to goal state

# execute skill
# do not reset environment after skill execution, but set actor to init z plane above its current position
_, plan = fm.breadth_first_search(init_state.flatten(), goal_state.flatten())
print(f"plan = {plan}")

#plan = [0, 1, 1, 1, 0, 1]
skill_idx = 0
obs, _ = env.reset(skill=plan[skill_idx], actor_pos=np.array([0., 0.]), sym_state_in=init_state)
num_steps = 0
while True:
    env.scene.C.view()
    env.scene.C.view_savePng('z.vid/')
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    num_steps += 1
    print(f"num_steps = {num_steps}")
    if terminated or num_steps > 100:
        # do not reset environment, but only set actor pos to init z-plane
        num_steps = 0
        skill_idx += 1
        if skill_idx >= len(plan):
            break
        print(skill_idx)
        obs = env.execution_reset(skill=plan[skill_idx])

time.sleep(10.)

del model
env.close()