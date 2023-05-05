import numpy as np
import time
from puzzle_scene import PuzzleScene
from robotic import ry
from gymframework.puzzle_env import PuzzleEnv
from gym.utils.env_checker import check_env


if __name__ == '__main__':

    # sanity check of custom env
    env = PuzzleEnv()
    ##check_env(env)
    env.scene.C.setJointState([-0.1, 0.1, 0.1, 0.])
    print("================================")
    print("setting simulation joints to: ", env.scene.C.getJointState())
    env.scene.S.setState(env.scene.C.getFrameState())
    env.scene.S.step( np.zeros(len(env.scene.q0)), env.scene.tau, ry.ControlMode.velocity)
    print("joint state after setting State: ", env.scene.C.getJointState())
    time.sleep(5.)
    env.scene.reset()
    time.sleep(10.)

    """
    # go from initial configuration behind left cube
    # first move to right x-y-position
    #some good old fashioned IK
    myScene.v = np.array([0., -0.028, 0., 0])
    #myScene.v = np.array([0.025, -0.028, 0., 0.])
    myScene.velocity_control(30)

    # then move to right z-position
    myScene.v = np.array([0., 0., -0.05, 0.])
    myScene.velocity_control(30)

    # go forward
    myScene.v = np.array([0., 0.04, 0., 0.])
    myScene.velocity_control(100)

    myScene.reset()
    time.sleep(20.)

    myScene.C = 0
    myScene.S = 0
    """

