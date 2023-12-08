import numpy as np
import time
from puzzle_scene import PuzzleScene
from robotic import ry
from gymframework.puzzle_env_2x2_skill_conditioned import PuzzleEnv
from gym.utils.env_checker import check_env


if __name__ == '__main__':

    # sanity check of custom env
    env = PuzzleEnv(path="slidingPuzzle_2x2.g", puzzlesize=[2, 2], verbose=1, num_skills=1)
    print(f"init sym obs = {env.scene.sym_state}")

    print("RESETTING")
    env.reset()

    print("env.box = ", env.box)
    box_pos = (env.scene.C.getFrame("box" + str(env.box)).getPosition()).copy()
    print("box_pos = ", box_pos)

    time.sleep(2)

    # always some y and z-offset because of the way the wedge and the boxes were placed
    opt = box_pos.copy()

    opt[2] -= 0.2
    print("opt[2] = ", opt[2])
    #opt[1] += env.offset / 2
    ## additional offset in x-direction and y-direction dependent on skill
    ## (which side do we want to push box from?)
    print(f"opt pos x-dir = {env.opt_pos_dir[env.skill, 0]}")
    print(f"opt pos y-dir = {env.opt_pos_dir[env.skill, 1]}")
    opt[0] += env.offset * (env.opt_pos_dir[env.skill, 0])
    opt[1] += env.offset * env.opt_pos_dir[env.skill, 1]
    # go to correct x-y-position
    #env.scene.q = np.array([opt[0], opt[1], opt[2], env.scene.q[3]])
    #print("positioned robot at optimal position")
    #env.scene.q = np.array([opt[0], opt[1], opt[2], env.scene.q[3]])
    env.scene.q = np.array([0.1, 0.1, 0.1, env.scene.q[3]])

    time.sleep(5)

    dist, _ = env.scene.C.eval(ry.FS.distance, ["box" + str(env.box), "wedge"])
    print("dist = ", dist)

    print("Gone to optimal position")
    time.sleep(5)

    print("sym_obs before push = ", env.scene.sym_state)
    # push box
    env.scene.v = np.array([0.2, 0., 0., 0.])

    env.scene.velocity_control(1000)
    dist, _ = env.scene.C.eval(ry.FS.distance, ["box" + str(env.box), "wedge"])

    print("dist after update = ", dist)

    print("sym_obs after push = ", env.scene.sym_state)

    #env.scene.v = np.array([-10.9, 0., 0., 0.])
    #env.scene.velocity_control(500)

    time.sleep(5.)
    #sym_obs = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
    #                    [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1]])
    #env.scene.sym_state = sym_obs
    #env.scene.set_to_symbolic_state()
    #print("Set new state")
    #time.sleep(10.)
    #check_env(env)
    #print("q0 = ", env.scene.q0)
    #env.scene.q = np.array([-0.2, 0.2, 0., 0.])
    #time.sleep(5.)
    #env.scene.v = np.array([-0.5, -0.5, 0., 0.])
    #env.scene.velocity_control(500)
    #time.sleep(5.)
    #env.scene.reset()
    #time.sleep(10.)

    # go from initial configuration behind left cube
    # first move to right x-y-position
    #some good old fashioned IK
    #myScene = PuzzleScene(filename="slidingPuzzle.g")
    #myScene.v = np.array([0., -0.028, 0., 0])
    #myScene.v = np.array([0., -0.7, 0., 0.])
    #myScene.velocity_control(150)
    #print("joint config = ", myScene.C.getJointState())

    # then move to right z-position
    #myScene.v = np.array([0., 0., -2., 0.])
    #myScene.velocity_control(300)

    # go forward
    #myScene.v = np.array([0., 0.5, 0., 0.])
    #myScene.velocity_control(100)
    #time.sleep(2.)

    #myScene.reset()

    #time.sleep(20.)

    #myScene.C = 0
    #myScene.S = 0

