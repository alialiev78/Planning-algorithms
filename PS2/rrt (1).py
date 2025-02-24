from typing import List, Callable
import numpy as np
from environment import State, ManipulatorEnv
from angle_util import *


class RRTPlanner:

    def __init__(self,
                 env: ManipulatorEnv,
                 distance_fn: Callable,
                 max_angle_step: float = 10.0):
        """
        :param env: manipulator environment
        :param distance_fn: function distance_fn(state1, state2) -> float
        :param max_angle_step: max allowed step for each joint in degrees
        """
        self._env = env
        self._distance_fn = distance_fn
        self._max_angle_step = max_angle_step

    def plan(self,
             start_state: State,
             goal_state: State) -> List[State]:
        N = 10000
        parent_table = {tuple(start_state.angles): None}
        nodes = [start_state.angles]
        for i in range(N):
            if np.random.uniform() < 0.1:
                q_rand = goal_state.angles
            else:
                q_rand = np.random.uniform(low = -180, high = 180, size = 4)
            state_nearest = nodes[np.argmin([self._distance_fn(q_rand, q) for q in nodes])]
            
            n = int((self._distance_fn(state_nearest, q_rand) // self._max_angle_step) + 1)
            steps = angle_linspace(state_nearest, q_rand, n)
            prev_state = State(state_nearest)
            for s in steps[1:]:
                current_state = State(s)
                if self._env.check_collision(current_state):
                    break
                else:
                    parent_table[tuple(current_state.angles)] = tuple(prev_state.angles)
                    nodes.append(current_state.angles)
                    prev_state = current_state

            if np.min([self._distance_fn(goal_state.angles, q) for q in nodes]) <= self._max_angle_step:
                index_nearest = np.argmin([self._distance_fn(goal_state.angles, q) for q in nodes])
                state_nearest = State(nodes[index_nearest])
                parent_table[tuple(goal_state.angles)] = tuple(state_nearest.angles)
                
                plan = []
                plan = list(reversed(get_plan(tuple(goal_state.angles), parent_table, plan, tuple(start_state.angles))))
                result = {}
                result['plan'] = plan
                result['iterations'] = i
                result['nodes_n'] = len(parent_table.keys())
                return result
                
        return 'YOU DIED'
    
    
    

def get_plan(x, parent_table, plan, x_i):
    if x == x_i:
        return plan
    else:
        child = parent_table[x]
        plan.append(State(np.array(child)))
        return get_plan(child, parent_table, plan, x_i)
    
