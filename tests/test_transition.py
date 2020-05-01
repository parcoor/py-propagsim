import numpy as np
from time import time


### Setup

def vectorized_choice(prob_matrix, axis=1):
    """ 
    selects index according to weights in `prob_matrix` rows (if `axis`==0), cols otherwise 
    see https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix
    """
    s = prob_matrix.cumsum(axis=axis)
    r = np.random.rand(prob_matrix.shape[1-axis]).reshape(2*(1-axis)-1, 2*axis - 1)
    k = (s < r).sum(axis=axis)
    return k

n_agents = 2000000
n_distinct_transitions = 15
n_states = 7

transitions = np.random.uniform(size=n_distinct_transitions * n_states ** 2)
transitions = transitions.reshape((n_states, n_states, n_distinct_transitions))
transitions /= np.linalg.norm(transitions, ord=1, axis=1, keepdims=True)

# !!! hint: for each transition (in depth) select only rows and columns corresponding to finite duration state (not "virgin", not dead, not recovered)
# transitions = transitions[ids_finite states, ids_finite_state,:]
# select also agents not in finite state

agent_transitions = np.random.choice(np.arange(0, n_distinct_transitions), size=n_agents)
agent_current_states = np.random.choice(np.arange(0, n_states), size=n_agents)

### Actual 
t0 = time()
# Reorder
order = np.argsort(agent_transitions)
agent_transitions = agent_transitions[order]
agent_current_states = agent_current_states[order]
# Select rows corresponding to transitions to do
transition_states = np.vstack((agent_transitions, agent_current_states))
unique_cols, inverse, counts = np.unique(transition_states, return_inverse=True, return_counts=True, axis=1)
transitions = transitions[unique_cols[1,:],:,unique_cols[0,:]]
# Repeat rows according to number of agents to draw for
transitions = np.repeat(transitions, counts, axis=0)
# Select new states according to transition matrix
new_states = vectorized_choice(transitions)
new_states = new_states[inverse]

print(f'duration: {time() - t0}')


print(f'agent_current_states:\n{agent_current_states}\nagent_current_states.shape: {agent_current_states.shape}')
print(f'new_states:\n{new_states}\nnew_states.shape: {new_states.shape}')


