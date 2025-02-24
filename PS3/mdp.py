import numpy as np
from utils import *


class MDP:

    def __init__(self,
                 env: Environment,
                 goal: tuple,
                 gamma: float = 0.99):
        """
        env is the grid enviroment
        goal is the goal state
        gamma is the discount factor
        """
        self._env = env
        self._goal = goal
        self._gamma = gamma
        self._V = np.zeros(env.shape)
        self._policy = np.zeros(self._env.shape, 'b') #type byte (or numpy.int8)

    def calculate_value_function(self):
        """
        This function uses the Value Iteration algorithm to fill in the
        optimal value function
        """
        num_iterations = 100
        for i in range(num_iterations):
            for j in range(self._env.shape[0]):
                for el in range(self._env.shape[1]):
                    state = (j, el)
                    if not self._env.state_consistency_check(state) or state == self._goal:
                        continue
                
                    value = -np.inf
                    for a in action_space:
                        state_propagated_list, prob_list = self._env.probabilistic_transition_function(state, a)
                        expected_value = 0

                        for l in range(len(state_propagated_list)):
                            next_state = state_propagated_list[l]
                            prob = prob_list[l]
                            if not self._env.state_consistency_check(next_state):
                                continue
                            reward = -1 if not self._env.state_consistency_check(next_state) else 0
                            if next_state == self._goal:
                                reward = 1
                            expected_value += prob*(reward + self._gamma*self._V[next_state])

                        value = max(expected_value, value)

                    self._V[state] = value
            
        return self._V

    def calculate_policy(self):
        """
        Only to be run AFTER Vopt has been calculated.
        
        output:
        policy: a map from each state s to the greedy best action a to execute
        """

        for j in range(self._env.shape[0]):
            for el in range(self._env.shape[1]):
                state = (j, el)
                if not self._env.state_consistency_check(state) or state == self._goal:
                    continue 

                best_action = None
                value = -np.inf

                for i, a in enumerate(action_space):
                    state_propagated_list, prob_list = self._env.probabilistic_transition_function(state, a)
                    expected_value = 0

                    for l in range(len(state_propagated_list)):
                        next_state = state_propagated_list[l]
                        prob = prob_list[l]
                        if not self._env.state_consistency_check(next_state):
                            continue
                        if not self._env.state_consistency_check(next_state):
                            reward = -1
                        else:
                            reward = 0
                        if next_state == self._goal:
                            reward = 1
                        expected_value += prob*(reward + self._gamma*self._V[next_state])

                    if value > max_value:
                        value = expected_value
                        action = i

                self._policy[state] = best_action 

        return self._policy

    def policy(self,state:tuple) -> int:
        """
        returns the action according to the policy
        """
        return self._policy[state]

