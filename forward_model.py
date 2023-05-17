import numpy as np
import torch
import torch.nn as nn
from collection import OrderedDict

"""
- The forward model is a tree, connecting all skills
- Probability of symbolic state being reached, when performing a skill in particular symbolic state 
(effect of skill executions)
- skill conditioned
- multilayer perceptron that predicts weather entry in symbolic observation will flip on skill execution
- trained using gradient descent (using ADAM) (given tuples of sym observation, skill and successor observation)
"""

"""
- skills are actions that lead to transition in symbolic observation
- forward model captures effect of skill on symbolic observation
"""


class ForwardModel:

    """
    - plan sequence of skills from start to goal symbolic state using breadth-first search

    - Model of distribution over z' (successor state)
    Probabilities of [(state, skill) -> succ_state]
    -> MLP predicts probabilities of entries in symbolic observation to flip (p_flip = f(z_0, k))
    -> returns most likely successor state given skill in current state

    Function successor:
    - state that is most probable given skill and model
    - node (board configuration) expansion on symbolic forward model (applying feasible actions)

    Breath-first search
    - begin at goal state
    - nodes through node expansion using successor function
    """
    def __init__(self,
                 width: int,
                 height: int,
                 num_skills: int):

        self.sym_obs_size = width * height * (widht * height -1)
        if torch.cuda.is_availabel():
            self.device = 'cuda:0'
        else
            self.device = 'cpu'

        self.model = nn.Sequential(OrderedDict[
                                       ('input', nn.Linear(self.sym_obs_size + num_skills, 256, device=self.device)),
                                       ('act1', nn.ReLu()),
                                       ('hidden1', nn.Linear(256, 256, device=self.device)),
                                       ('act2', nn.ReLU()),
                                       ('output', nn.Linear(256, self.sym_obs_size, device=self.device))
                                   ])

        # loss: negative log-likelihood - log q(z_T | z_0, k)
        self.loss =
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def successor(self, state, skill):
        """
        Returns successor nodes of a given node
        Args:
            :param state: node to find successor to
            :param skill: skill to apply
        Returns:
            succ: most likely successor when applying skill in state
        """
        # give input of state, skill pear to mlp and translate output to new state

        raise NotImplementedError

    def feasible(self, state):
        """
        Args:
            :param state: symbolic state
        Returns:
            K: all skills that are applicable in input state
        """
        # problem: for that we need prediction what skill will do
        # thus mlp must be able to handle inputs of state, skill pears where skill is not applicable in that state
        # maybe not a problem after all, because either the forward model will learn that no state change occurs,
        # or that successor state is invalid
        # in both cases we can ignore this state transition in the breadth first search
        raise NotImplementedError

    def _backtrace(self, start, goal, parent, skill):
        path = goal
        skills = []
        while not (path[-1] == start).all():
            path.append(parent[path[-1]])
            skills.append(skill[path[-1]])

        return path.reverse(), skills.reverse()

    def breadth_first_search(self, start, goal):
        """
        Returns sequence of skills to go from start to goal configuration
        as predicted by current forward_model
        Args:
            :param start: start configuration (symbolic state)
            :param goal: goal configuration (symbolic state)
        Returns:
            state_sequence: sequence of symbolic states the forward model predicts ill be visited when executing the predicted skills
            skill_sequence: sequence of skills to be executed to go from start to goal configuration
            depth: solution depth
        """
        visited = []
        queue = []
        parent = {}
        skill = {}

        # start search from start configuration
        visited.append(start)
        queue.append([start])

        while queue:
            # go through queue
            state = queue.pop(0)

            # node expansion through applying feasible actions
            for k in self.feasible(state):
                # find successor state
                next_state = self.successor(state, k)
                # record skill used to get to next_state
                skill[next_state] = k
                # record parent
                parent[next_state] = state

                # if successor state is goal: break
                if (next_state == goal).all():
                    # get the state transitions and skills that lead to the goal
                    state_sequence, skill_sequence = self._backtrace(start, goal, parent, skill)
                    return state_sequence, skill_sequence
                # look if successor was already visited
                for node in visited:
                    if (next_state == node).all():
                        break
                # if not append successor to visited and queue
                visited.append(next_state)
                queue.append(next_state)

        raise Error("Goal Not reachable from start configuration")