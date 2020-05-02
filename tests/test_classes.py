from classes import State, Agent, Cell, Transitions, Map
import numpy as np
from time import time


def get_alpha_beta(min_value, max_value, mean_value):
    """ for the duration on a state, draw from a beta distribution with parameter alpha and beta """
    x = (mean_value - min_value) / (max_value - min_value)
    z = 1 / x - 1
    a, b = 2, 2 * z
    return a, b


def draw_beta(min_value, max_value, mean_value, n_values, round=False):
    """ draw `n_values` values between `min_value` and `max_value` having 
    `mean_value` as (asymptotical) average"""
    a, b = get_alpha_beta(min_value, max_value, mean_value)
    durations = np.random.beta(a, b, n_values) * (max_value - min_value) + min_value
    if round:
        durations = np.around(durations)
    return durations.reshape(-1, 1)

# =========== States ==============

state0 = State(id=0, name='healthy', contagiousity=0, sensitivity=1, severity=0)
state1 = State(id=1, name='asymptomatic', contagiousity=.8, sensitivity=0, severity=0.1)
state2 = State(id=2, name='mild', contagiousity=.6, sensitivity=0, severity=0.8)
state3 = State(id=3, name='hospital', contagiousity=.1, sensitivity=0, severity=1)
state4 = State(id=4, name='reanimation', contagiousity=.05, sensitivity=0, severity=1)
state5 = State(id=5, name='dead', contagiousity=0, sensitivity=0, severity=1)
state6 = State(id=6, name='recovered', contagiousity=0, sensitivity=0, severity=0)

states = [state0, state1, state2, state3, state4, state5, state6]
id2state = {state.get_id(): state.get_name() for state in states}

# ========= Transitions ==========

# For people younger than 15yo
transitions_15 = Transitions(0, np.array([[1, 0, 0, 0, 0, 0, 0], 
                                     [0, 0, 0.5, 0, 0, 0, 0.5], 
                                     [0, 0, 0, 0.3, 0, 0, 0.7],
                                     [0, 0, 0, 0, 0.3, 0, 0.7],
                                     [0, 0, 0, 0, 0, 0.5, 0.5],
                                     [0, 0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 0, 1]]))

# For people younger between 15 and 44yo
transitions_15_44 = Transitions(1, np.array([[1, 0, 0, 0, 0, 0, 0], 
                                         [0, 0, 0.5, 0, 0, 0, 0.5], 
                                         [0, 0, 0, 0.3, 0, 0, 0.7],
                                         [0, 0, 0, 0, 0.3, 0, 0.7],
                                         [0, 0, 0, 0, 0, 0.5, 0.5],
                                         [0, 0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 0, 1]]))

# For people younger between 45 and 64yo
transitions_45_64 = Transitions(2, np.array([[1, 0, 0, 0, 0, 0, 0], 
                                         [0, 0, 0.5, 0, 0, 0, 0.5], 
                                         [0, 0, 0, 0.3, 0, 0, 0.7],
                                         [0, 0, 0, 0, 0.3, 0, 0.7],
                                         [0, 0, 0, 0, 0, 0.5, 0.5],
                                         [0, 0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 0, 1]]))

# For people younger between 65 and 75yo
transitions_65_74 = Transitions(3, np.array([[1, 0, 0, 0, 0, 0, 0], 
                                         [0, 0, 0.5, 0, 0, 0, 0.5], 
                                         [0, 0, 0, 0.3, 0, 0, 0.7],
                                         [0, 0, 0, 0, 0.3, 0, 0.7],
                                         [0, 0, 0, 0, 0, 0.5, 0.5],
                                         [0, 0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 0, 1]]))

# For people younger >= 75yo
transitions_75 = Transitions(4, np.array([[1, 0, 0, 0, 0, 0, 0], 
                                     [0, 0, 0.5, 0, 0, 0, 0.5], 
                                     [0, 0, 0, 0.3, 0, 0, 0.7],
                                     [0, 0, 0, 0, 0.3, 0, 0.7],
                                     [0, 0, 0, 0, 0, 0.5, 0.5],
                                     [0, 0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 0, 1]]))

transitions = [transitions_15, transitions_15_44, transitions_45_64, transitions_65_74, transitions_75]



# =========== Agents ================

N_AGENTS = 1000

prop_population = [.17, .35, .3, .1, .08]
draw_transitions = np.random.choice(range(len(transitions)), N_AGENTS, p=prop_population)

AVG_AGENTS_HOME = 2.2

n_home_cells = int(N_AGENTS / AVG_AGENTS_HOME)
draw_home_cells = np.random.choice(range(n_home_cells), N_AGENTS)
p_moves = draw_beta(0, 1, .5, N_AGENTS)

durations_healthy = durations_dead = durations_recovered = np.ones(shape=(N_AGENTS, 1)) * -1
durations_asymptomatic = draw_beta(1, 14, 5, N_AGENTS, True)
durations_mild = draw_beta(5, 10, 7, N_AGENTS, True)
durations_hospital = draw_beta(1, 8, 4, N_AGENTS, True)
durations_reanimation = draw_beta(15, 30, 21, N_AGENTS, True)

durations = [durations_healthy, durations_asymptomatic, durations_mild, 
             durations_hospital, durations_reanimation, durations_dead, durations_recovered]

durations = np.concatenate(durations, axis=1)

agents = []
for i in range(N_AGENTS):
    agent = Agent(id=i, p_move=p_moves[i], 
                  transitions=transitions[draw_transitions[i]], 
                  states=states, 
                  durations=durations[i,:].flatten(), 
                  current_state=state0, 
                  home_cell_id=draw_home_cells[i])
    agents.append(agent)


# ========== Cells ==============

N_CELLS = int(1.1 * n_home_cells)

positions_x = np.random.uniform(low=0, high=10, size=N_CELLS).reshape(-1, 1)
positions_y = np.random.uniform(low=0, high=10, size=N_CELLS).reshape(-1, 1)
positions = np.concatenate([positions_x, positions_y], axis=1)

attractivities = np.random.uniform(size=N_CELLS)
attractivities[:n_home_cells] = 0

unsafeties = np.random.uniform(size=N_CELLS)
unsafeties[:n_home_cells] = 1

cells = []
for i in range(N_CELLS):
    cell = Cell(id=i, 
                position=positions[i,:].flatten(), 
                attractivity=attractivities[i], 
                unsafety=unsafeties[i])
    cells.append(cell)


# =========== Map =============

map = Map(cells, agents, states, verbose=2)

n_infected_agents_start = 100
infected_agent_id = np.random.choice(range(N_AGENTS), size=n_infected_agents_start, replace=False)
new_state_id = 1

print(f'Injecting {n_infected_agents_start} contaminated agents out of {N_AGENTS} in map')

map.change_state_agents(np.array([infected_agent_id]), np.array([new_state_id]))

N_PERIODS = 30
N_MOVES_PER_PERIOD = 3

stats = {}
t_start = time()
for i in range(N_PERIODS):
    print(f'starting period {i}...')
    t0 = time()
    for j in range(N_MOVES_PER_PERIOD):
        t_ = time()
        map.make_move()
    map.forward_all_cells()
    states_ids, state_numbers = map.get_states_numbers()
    stats[i] = {states_ids[k]: state_numbers[k] for k in range(len(states_ids))}
    print(f'period {i} computed in {time() - t0}s')

print(f'duration: {time() - t_start}s')
print(stats)

print(f'r_factors: {map.get_r_factors()}')

infecting_agents, infected_agents, infected_periods = map.get_contamination_chain()
print(f'All infecting != infected? {(infecting_agents == infected_agents).sum() == 0}')