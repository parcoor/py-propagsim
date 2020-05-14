from classes import State, Agent, Cell, Transitions, Map
import numpy as np
from time import time


N_PERIODS = 30
N_MOVES_PER_PERIOD = 10
AVG_P_MOVE = .5 / N_MOVES_PER_PERIOD
N_AGENTS = 700000
N_INFECTED_AGENTS_START = int(N_AGENTS / 40)
N_SQUARES_AXIS = 30
AVG_AGENTS_HOME = 2.2
N_HOME_CELLS = int(N_AGENTS / AVG_AGENTS_HOME)
PROP_PUBLIC_CELLS = 1 / 70  # there is one public place for 70 people in France
N_CELLS = int(N_HOME_CELLS + N_AGENTS * PROP_PUBLIC_CELLS)
DSCALE = 30
AVG_UNSAFETY = .5


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
<<<<<<< HEAD
state1 = State(id=1, name='asymptomatic', contagiousity=.2, sensitivity=0, severity=0.1)
=======
state1 = State(id=1, name='asymptomatic', contagiousity=.9, sensitivity=0, severity=0.1)
>>>>>>> 024791b60731bd81bf57a6c52f3f58c77cab4579
state2 = State(id=2, name='mild', contagiousity=.8, sensitivity=0, severity=0.8)
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


prop_population = [.17, .35, .3, .1, .08]
draw_transitions = np.random.choice(range(len(transitions)), N_AGENTS, p=prop_population)

draw_home_cells = np.random.choice(range(N_HOME_CELLS), N_AGENTS)


p_moves = draw_beta(0, 1, AVG_P_MOVE, N_AGENTS)

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



positions_x = np.random.uniform(low=0, high=N_SQUARES_AXIS, size=N_CELLS).reshape(-1, 1)
positions_y = np.random.uniform(low=0, high=N_SQUARES_AXIS, size=N_CELLS).reshape(-1, 1)
positions = np.concatenate([positions_x, positions_y], axis=1)

attractivities = np.random.uniform(size=N_CELLS)
attractivities[:N_HOME_CELLS] = 0

unsafeties = draw_beta(0, 1, AVG_UNSAFETY, N_CELLS).flatten()
unsafeties[:N_HOME_CELLS] = 1

cells = []
for i in range(N_CELLS):
    cell = Cell(id=i, 
                position=positions[i,:].flatten(), 
                attractivity=attractivities[i], 
                unsafety=unsafeties[i])
    cells.append(cell)


# =========== Map =============

map = Map(cells, agents, states, dscale=DSCALE, verbose=0)


infected_agent_id = np.random.choice(range(N_AGENTS), size=N_INFECTED_AGENTS_START, replace=False)
new_state_id = 1

print(f'Injecting {N_INFECTED_AGENTS_START} contaminated agents out of {N_AGENTS} in map')

map.change_state_agents(np.array([infected_agent_id]), np.array([new_state_id]))

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
    if 5 in stats:
        print()
    print(f'period {i} computed in {time() - t0}s')

print(f'duration: {time() - t_start}s')
print(stats)
