import numpy as np
import time
from puzzle_scene import PuzzleScene
from robotic import ry
from gymframework.puzzle_env_small_skill_conditioned import PuzzleEnv
from gym.utils.env_checker import check_env


if __name__ == '__main__':

    # sanity check of custom env
    env = PuzzleEnv(path="Puzzles/slidingPuzzle_1x2.g", verbose=1:wq)

    env.reset()

    print(f"init sym obs = {env.scene.sym_state}")

    box_pos = (env.scene.C.getFrame("box" + str(env.box)).getPosition()).copy()
    print("init box pos = ", box_pos)
    # always some y and z-offset because of the way the wedge and the boxes were placed
    opt = box_pos.copy()
    opt[2] -= 0.2
    opt[1] -= env.offset / 2
    # additional offset in x-direction and y-direction dependent on skill
    # (which side do we want to push box from?)
    opt[0] += env.offset * env.opt_pos_dir[env.skill, 0]
    opt[1] += env.offset * env.opt_pos_dir[env.skill, 1]
    time.sleep(5)
    env.scene.q = np.array([opt[0], opt[1], opt[2], env.scene.q[3]])
    print("init q = ", env.scene.q)
    dist, _ = env.scene.C.eval(ry.FS.distance, ["box" + str(env.box), "wedge"])
    print("dist = ", dist)
    print("=========================================")
    if box_pos[0] > 0:
        v = -0.2
    else:
        v = 0.2
    time.sleep(5)
    for i in range(100):
        env.scene.v = np.array([v, 0., 0., 0.])
        env.scene.velocity_control(10)
        print("q = ", env.scene.q)
        dist , _ = env.scene.C.eval(ry.FS.distance, ["box" + str(env.box), "wedge"])
        print("dist = ", dist)
        box_pos = (env.scene.C.getFrame("box" + str(env.box)).getPosition()).copy()
        print("box_pos = ", box_pos)
        print("=================================================")
    print("sym_obs after push = ", env.scene.sym_state)

    time.sleep(20)
    # push box
    env.scene.v = np.array([0.2, 0., 0., 0.])

    env.scene.velocity_control(1000)
