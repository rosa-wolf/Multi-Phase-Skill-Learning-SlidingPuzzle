#from robotic import ry
import robotic as ry
import numpy as np
import time

import os

class PuzzleScene:
    def __init__(self,
                 filename: str,
                 puzzlesize=[2, 3],
                 lim=np.array([0.25, 0.25, 0.25]),
                 snapRatio = 4,
                 verbose=0):
        """

        field with discrete places
        _____________
        | 0 | 1 | 2 |
        | 3 | 4 | 5 |
        ------------
        Initially box0 is on place 0, box1 on place 1 etc. and place 5 is empty
        Thus, the initial symbolic observation is:

          | 0 | 1 | 2 | 3 | 4 | 5 |
        0 | 1 | 0 | 0 | 0 | 0 | 0 |
        1 | 0 | 1 | 0 | 0 | 0 | 0 |
        2 | 0 | 0 | 1 | 0 | 0 | 0 |
        3 | 0 | 0 | 0 | 1 | 0 | 0 |
        4 | 0 | 0 | 0 | 0 | 1 | 0 |

        :type   filename:
                puzzlesize: size of the puzzle (default is 2x3)



        """
        ry.params_add({'physx/motorKp': 1000., 'physx/motorKd': 100., 'physx/defaultFriction': 0.05})  # 1000 & 100 & 0.05

        self.C = ry.Config()
        self.C.addFile(filename)
        # store initial configuration
        self._X0 = self.C.getFrameState()
        self.C.setFrameState(self.X0)

        # initialize simulation
        self.verbose = verbose
        self.S = ry.Simulation(self.C, ry.SimulationEngine.physx, self.verbose)

        self.puzzlesize = puzzlesize
        self.pieces = self.puzzlesize[0] * self.puzzlesize[1] - 1

        # joint limits (x, y, z) limits
        self.q_lim = np.array([[-lim[0], lim[0]], [-lim[1], lim[1]], [-lim[2], lim[2]]])

        # delta t
        self.tau = .01

        # get initial orientation of puzzle pieces (assumtion: all pieces have same orientation)
        self.quat0 = self.C.getFrame("box0").getQuaternion()
        # get initial positions of puzzle pieces to get discrete positions for symbolic state
        # go through all puzzlepieces
        # store position of each discrete place that exists in the symbolic state
        self.discrete_pos = np.empty((self.puzzlesize[0] * self.puzzlesize[1], 3))
        # initialize symbolic state
        self._sym_state = np.zeros((self.pieces, self.pieces + 1), dtype=int)

        for i in range(self.pieces):
            name = "box" + str(i)
            self.discrete_pos[i] = self.C.getFrame(name).getPosition()
            self._sym_state[i, i] = 1

        # beware: doesn't fit for the trained policies of the 1x2 puzzle
        # determine position of place which is initially empty
        # !!! Assumption: Always field with highest index is initially empty
        # and center of all fields have distance of 0.1 in all dimensions !!!
        self.discrete_pos[-1, 0] = self.discrete_pos[-2, 0] - 0.1
        self.discrete_pos[-1, 1:] = self.discrete_pos[-2, 1:]

        # store intial symbolic state
        self.sym_state0 = self.sym_state.copy()

        # radius for "snapping" (just any value for now)
        dist = np.linalg.norm(self.discrete_pos[0] - self.discrete_pos[1])
        self.snapRad = dist/snapRatio # must be smaller or equal to dist/2.

        # initialize state (q, \dot q)
        # variable for joint configuration
        self._q = self.C.getJointState()
        # store initial joint configuration to be able to reset to it later
        self._q0 = self._q.copy()
        self._v = np.zeros(len(self.q0))
        self._state = self._q, self._v, self._sym_state

    @property
    def state(self) -> tuple:
        # update state
        self._q = self.C.getJointState()
        return self._q, self._v, self._sym_state

    @property
    def sym_state(self) -> np.ndarray:
        return self._sym_state

    @sym_state.setter
    def sym_state(self, value: np.ndarray):
        self._sym_state = value

    @property
    def q(self) -> np.ndarray:
        # update q
        self._q = self.C.getJointState()
        return self._q

    @q.setter
    def q(self, value: np.ndarray):
        self.C.setJointState(value)
        self.S.setState(self.C.getFrameState())
        self.S.step([], self.tau, ry.ControlMode.none)
        self._q = value

    @property
    def q0(self) -> np.ndarray:
        return self._q0

    @property
    def X0(self):
        return self._X0

    @property
    def v(self) -> np.ndarray:
        """
        Return the current velocity
        """
        return self._v

    @v.setter
    def v(self, value: np.ndarray) -> None:
        self._v = value

    def velocity_control(self, n) -> bool:
        """
        Performs IK velocity control for n steps
        :param n: number of steps in velocity control
        :return: true if symbolic state changed
        """
        new_state = False
        for i in range(n):
            self.S.step(self.v, self.tau, ry.ControlMode.velocity)
            new_state = self.update_symbolic_state()
            # stop movement if symbolic state has changed
            if new_state:
                for _ in range(15):
                    self.S.step(self.v, self.tau, ry.ControlMode.velocity)
                self.update_symbolic_state()
                # set new symbolic state in simulation
                self.set_to_symbolic_state()
                self.v = np.zeros(len(self.q0))
                return new_state
        # set v to zero when movement is finished
        self.v = np.zeros(len(self.q0))
        return new_state

    def reset(self) -> None:
        """
        resets whole scene to initial state
        """
        self.sym_state = self.sym_state0.copy()
        self.v = np.zeros(len(self.q0))

        # set robot back to initial configuration
        # pass state on to simulation
        self.C.setJointState(self.q0)
        self.C.setFrameState(self.X0)
        del self.S
        self.S = ry.Simulation(self.C, ry.SimulationEngine.physx, self.verbose)

    def check_limits(self) -> bool:
        """
        Sets joints to limits if current joint  values exceed limits
        Returns True if joint values lie within joint limits

        """
        self._q = self.C.getJointState()
        new_q = self.q.copy()
        new_q[3] = self.q0[3]

        in_limit = True

        if self._q[0] < self.q_lim[0, 0]:
            new_q[0] = self.q_lim[0, 0]
            in_limit = False
        elif self._q[0] > self.q_lim[0, 1]:
            new_q[0] = self.q_lim[0, 1]
            in_limit = False

        if self._q[1] < self.q_lim[1, 0]:
            new_q[1] = self.q_lim[1, 0]
            in_limit = False
        elif self._q[1] > self.q_lim[1, 1]:
            new_q[1] = self.q_lim[1, 1]
            in_limit = False

        if self._q[2] < self.q_lim[2, 0]:
            new_q[2] = self.q_lim[2, 0]
            in_limit = False
        elif self._q[2] > self.q_lim[2, 1]:
            new_q[2] = self.q_lim[2, 1]
            in_limit = False

        if not in_limit:
            self.q = new_q
            print("q after setback = ", self.q)

        return in_limit

    def set_board_state(self, state) -> None:
        """
        Transforms a continuous state of the puzzle pieces to discrete positions

        :param state: symbolic state of the board game
        """
        raise NotImplementedError

    def get_positions(self) -> [float]:
        """
        Retrieves the current positions of the puzzle pieces in the simulation
        :return: the positions of all puzzle pieces
        """

        pos = []
        for i in range(self.pieces):
            name = "box" + str(i)
            pos.append(self.C.getFrame(name).getPosition())

        return np.array(pos)

    def set_to_symbolic_state(self, hard=False, zero_vel=False) -> None:
        """
         (Snapping) Sets position of all puzzle pieces to the current symbolic state
        :param zero_vel: whether to set the velocities of the actor to zero
        :param hard: if true we reinitialize the simulation, else we update existing simulation
        :return: None
        """
        # if new state is invalid don't set to the new state
        if not self.valid_state():
            return

        # set all pieces to their current symbolic state positions
        for i in range(self.pieces):
            name = "box" + str(i)
            # get position of boxes in current symbolic state
            idx = np.where(self._sym_state[i] == 1)[0]
            pos = self.discrete_pos[idx]
            # set position and orientation of boxes in Config
            self.C.getFrame(name).setQuaternion(self.quat0)
            self.C.getFrame(name).setPosition(pos)

        if hard:
            # reinitialize simulation
            del self.S
            self.S = ry.Simulation(self.C, ry.SimulationEngine.physx, self.verbose)
        else:
            self.S.pushConfigurationToSimulator()
            self.S.step([], self.tau, ry.ControlMode.none)

    def valid_state(self) -> bool:
        """
        Checks whether a symbolic state is valid
        :return: True, if state is valid
        """
        # Check that symbolic state is valid
        # i.e. that sum of each row and each column but one is 1 and that all values are 1 or 0
        col_sum = np.sum(self._sym_state, axis=0)
        row_sum = np.sum(self._sym_state, axis=1)

        if np.sum(col_sum) != self.pieces or np.sum(row_sum) != self.pieces:
            return False
        if np.any(col_sum > 1) or np.any(col_sum < 0) or np.any(row_sum > 1) or np.any(row_sum < 0):
            return False

        return True

    def update_symbolic_state(self) -> bool:
        """
        Detects whether symbolic state changed (based on snapping radius)
        and if it changed sets the new symbolic states
        :return: true if symbolic state changed
        """
        changed = False
        prev_state = self.sym_state.copy()

        # check if position ob puzzle piece is in radius of symbolic state position
        positions = self.get_positions()
        # go through all puzzle pieces
        for i in range(self.pieces):
            # check whether old symbolic state changes
            old_state = np.where(self.sym_state[i] == 1)[0]
            # go through all states except old one and look if puzzle piece is near enough to invoke change of
            # symbolic state
            for j in range(self.pieces + 1):
                if j != old_state:
                    if np.linalg.norm(positions[i] - self.discrete_pos[j]) <= self.snapRad:
                        # symbolic state changes
                        changed = True
                        self.sym_state[i, old_state] = 0
                        self.sym_state[i, j] = 1

        return changed
    

