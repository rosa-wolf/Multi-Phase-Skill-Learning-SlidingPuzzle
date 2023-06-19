from robotic import ry
import numpy as np
import time


class PuzzleScene:
    def __init__(self,
                 filename: str,
                 puzzlesize=None,
                 verbose=0):
        """

        field with discrete places
        _____________
        | 3 | 4 | 5 |
        | 0 | 1 | 2 |
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
        ry.params_add({'physx/motorKp': 1000., 'physx/motorKd': 100., 'physx/defaultFriction': 0.1})  # 1000 & 100

        #initialize configuration
        if puzzlesize is None:
            puzzlesize = [2, 3]
        self.puzzlesize = puzzlesize
        self.pieces = self.puzzlesize[0] * self.puzzlesize[1] - 1

        self.C = ry.Config()
        self.C.addFile(filename)

        # TODO: don't hardcode joint limits
        # joint limits (x, y, z) limits
        self.q_lim = np.array([[-.2, .2], [-.2, .2], [-.45, 0.]])
        #self.X0 = self.C.getFrameState()
        #self.C.setFrameState(self.X0)  # why do we need this? Setting feature with same values it already has?

        # store initial configuration
        self._X0 = self.C.getFrameState()
        self.C.setFrameState(self.X0)

        # initialize simulation
        self.verbose = verbose
        self.S = ry.Simulation(self.C, ry.SimulatorEngine.physx, self.verbose)

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
        # determine position of place which is initially empty (is not )
        self.discrete_pos[-1, 0] = self.discrete_pos[4, 0] + np.abs(self.discrete_pos[4, 0] - self.discrete_pos[3, 0])
        self.discrete_pos[-1, 1:] = self.discrete_pos[4, 1:]

        # store intial symbolic state
        self.sym_state0 = self.sym_state.copy()

        # radius for "snapping" (just any value for now)
        dist = np.linalg.norm(self.discrete_pos[0] - self.discrete_pos[1])
        self.snapRad = dist/4. # must be smaller or equal to dist/2.

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
        self.S = ry.Simulation(self.C, ry.SimulatorEngine.physx, self.verbose)
        #self.S.pushConfigurationToSimulator()
        # set blocks back to original positions
        #self.S.step([], self.tau,  ry.ControlMode.none)
        #self.set_to_symbolic_state()

    def check_limits(self) -> bool:
        """
        Returns True if joint values lie within joint limits
        """
        self._q = self.C.getJointState()
        new_q = self.q.copy()
        new_q[2] = self.q0[2]
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

        if not in_limit:
            self.q = new_q

        #if ((self._q[:3] >= self.q_lim[:, 0]) & (self._q[:3] <= self.q_lim[:, 1])).all():
        #    return True
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

    def set_to_symbolic_state(self) -> None:
        """
         (Snapping) Sets position of all puzzle pieces to the current symbolic state
        :return: None
        """
        # if new state is invalid don't set to the new state
        if not self.valid_state():
            return

        # set all pieces to their current symbolic state positions
        for i in range(self.pieces):
            name = "box" + str(i)
            # get position of box in current symbolic state
            idx = np.where(self._sym_state[i] == 1)[0]
            pos = self.discrete_pos[idx]
            # set position and orientation of box in Config
            self.C.getFrame(name).setQuaternion(self.quat0)
            self.C.getFrame(name).setPosition(pos)
            #new_pos = self.C.getFrame(name).getPosition()
        # pass state on to simulation
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
            # go through all states but except old one and look if puzzle piece is near enough to invoke change of
            # symbolic state
            for j in range(self.pieces + 1):
                if j != old_state:
                    if np.linalg.norm(positions[i] - self.discrete_pos[j]) <= self.snapRad:
                        # symbolic state changes
                        changed = True
                        self.sym_state[i, old_state] = 0
                        self.sym_state[i, j] = 1

        return changed
    

