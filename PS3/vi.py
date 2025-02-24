import numpy as np
from utils import *

class VI:
    def __init__(self, env: Environment, goal: tuple):
        """
        env is the grid environment, as defined in utils
        goal is the goal state
        """
        self._env = env
        self._goal = goal
        self._G = np.ones(self._env.shape) * np.inf  # Initialize cost-to-go
        self._policy = np.zeros(self._env.shape, 'b')  # Policy map
    
    
    def calculate_value_function(self):
        self._G[self._goal] = 0  
        num_iterations = 100
        delta = 1e-3
        for i in range(num_iterations):
            G_prev = self._G

            for j in range(self._env.shape[0]):
                for el in range(self._env.shape[1]):
                    state = (j, el)
                    if not self._env.state_consistency_check(state):
                        continue
                    costs = []                    
                    for a in action_space:
                        next_state, tmp = self._env.transition_function(state, a)
                        if tmp == True:
                            costs.append(1 + self._G[next_state])
                    if costs == True:
                        self._G[state] = min(min(costs), self._G[state])
            if np.max(np.abs(self._G - G_prev)) < delta:
                break
        return self._G

    
    def calculate_policy(self):
        """
        Calculate the optimal policy for the environment.

        Returns:
            dict: A map from each state (x, y) to the best action.
        """
        self._policy[self._goal] = -1 

        for j in range(self._env.shape[0]):
            for el in range(self._env.shape[1]):
                state = (j, el)

                if not self._env.state_consistency_check(state) or state == self._goal:
                    self._policy[state] = -1
                    continue

                min_cost = np.inf
                action = 'lalala'

                for a in action_space:
                    next_state, tmp = self._env.transition_function(state, a)
                    if tmp == True and (min_cost > 1 + self._G[next_state]):
                        min_cost = 1 + self._G[next_state]
                        action = a
                if action != 'lalala':
                    self._policy[state] = action_space.index(action)

        return self._policy
    

    def policy(self, state: tuple) -> int:
        """
        Retrieves the optimal action for the given state.
        """
        return self._policy[state]


