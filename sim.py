import numpy as np
import time
from puzzle_scene import PuzzleScene
from robotic import ry
from gymframework.puzzle_env_simple_old import PuzzleEnv
from gym.utils.env_checker import check_env


if __name__ == '__main__':

    # sanity check of custom env
    skill = 0
    env = PuzzleEnv(path="slidingPuzzle_1x2.g", puzzlesize=[1, 2], verbose=1, skill=skill, penalize=False)
    env.skill = skill
    print("sym_obs before push = ", env.scene.sym_state)

    # check position of maximal distance to opt
    opt = (env.scene.C.getFrame("box" + str(env.box)).getPosition()).copy()
    # always some y and z-offset because of the way the wedge and the boxes were placed
    opt[2] -= 0.25
    opt[1] -= env.offset / 2
    # additional offset in x-direction and y-direction dependent on skill
    # (which side do we want to push box from?)
    opt[0] += env.offset * env.opt_pos_dir[env.skill, 0]
    opt[1] += env.offset * env.opt_pos_dir[env.skill, 1]
    # go to correct x-y-position
    #env.scene.q = np.array([opt[0], opt[1], opt[2], env.scene.q[3]])
    #print("positioned robot at optimal position")
    env.scene.q = np.array([opt[0] + 0.02, opt[1] + 0.02, opt[2], env.scene.q[3] + 0.07])
    time.sleep(3)

    # push box
    env.scene.v = np.array([-0.2, 0., 0., 0.])
    env.scene.velocity_control(1000)

    print("sym_obs after push = ", env.scene.sym_state)

    env.scene.v = np.array([-10.9, 0., 0., 0.])
    env.scene.velocity_control(500)

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

