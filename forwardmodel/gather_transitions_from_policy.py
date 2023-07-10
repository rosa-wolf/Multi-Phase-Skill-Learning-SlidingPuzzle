
from ..gymframework.single_policy_env import PuzzleEnv

if __name__ == "__main__":
    # Environment
    env = PuzzleEnv(path='../SEADS_SlidingPuzzle/slidingPuzzle.g', max_steps=args.num_steps, random_init_pos=True,
                    random_init_config=True, evaluate=True)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    # agent is skill conditioned and thus also gets one-hot encoding of skill in addition to observation
    agent = SAC(env.observation_space.shape[0] + args.num_skills, env.action_space, args)

    path = "checkpoints_sparsereward/skill_conditioned_policy"
