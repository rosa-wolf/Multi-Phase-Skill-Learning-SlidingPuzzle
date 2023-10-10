import numpy as np
import time
from puzzle_scene import PuzzleScene
from robotic import ry
from gymframework.puzzle_env import PuzzleEnv
from gym.utils.env_checker import check_env


if __name__ == '__main__':

    # sanity check of custom env
    skill = 13
    env = PuzzleEnv(verbose=1, skill=skill, penalize=False)
    env.skill = skill

    field = env.skills[env.skill][0]
    # get box that is currently on that field
    box = np.where(env.scene.sym_state[:, field] == 1)[0][0]
    # read out position of box that should be pushed
    opt = (env.scene.C.getFrame("box" + str(box)).getPosition()).copy()
    print(opt[0])
    print(opt[1])
    print(opt[2])
    # offset in x-direction and y-direction
    print(box)
    print(env.offset)
    print(env.opt_pos_dir[env.skill, 0])
    # always z-offset of -0.3
    opt[2] -= 0.3
    # always y-offset
    opt[1] -= env.offset/2
    opt[0] += env.offset * env.opt_pos_dir[env.skill, 0]
    opt[1] += env.offset * env.opt_pos_dir[env.skill, 1]
    # go to correct x-y-position
    env.scene.q = np.array([opt[0], opt[1], opt[2], env.scene.q[3]])

    time.sleep(10)
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

