from classes import State, Agent, Cell, Transitions, Map
import cupy as cp
from time import time


N_PERIODS = 10
N_MOVES_PER_PERIOD = 4
AVG_P_MOVE = .5 / N_MOVES_PER_PERIOD
N_AGENTS = 700000
PROP_INFECTED_AGENTS_START = 1 / 400
N_SQUARES_AXIS = 200
AVG_AGENTS_HOME = 2.2
N_HOME_CELLS = int(N_AGENTS / AVG_AGENTS_HOME)
PROP_PUBLIC_CELLS = 1 / 70  # there is one public place for 70 people in France
N_CELLS = int(N_HOME_CELLS + N_AGENTS * PROP_PUBLIC_CELLS)
DSCALE = 0.1
AVG_UNSAFETY = .5

with cp.cuda.Device(0):

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
        durations = cp.random.beta(a, b, n_values) * (max_value - min_value) + min_value
        if round:
            durations = cp.around(durations)
        return durations.reshape(-1, 1).astype(cp.float16)

    states = ['healthy', 'asymptomatic', 'mild', 'hosp', 'reanimation', 'dead', 'recovered']
    id2state = {i: state for i, state in enumerate(states)}

    # ========= Transitions ==========

    # For people younger than 15yo
    transitions_15 = cp.array([[1, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0.5, 0, 0, 0, 0.5], 
                                [0, 0, 0, 0.3, 0, 0, 0.7],
                                [0, 0, 0, 0, 0.3, 0, 0.7],
                                [0, 0, 0, 0, 0, 0.5, 0.5],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1]])

    # For people younger between 15 and 44yo
    transitions_15_44 = cp.array([[1, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0.5, 0, 0, 0, 0.5], 
                                [0, 0, 0, 0.3, 0, 0, 0.7],
                                [0, 0, 0, 0, 0.3, 0, 0.7],
                                [0, 0, 0, 0, 0, 0.5, 0.5],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1]])

    # For people younger between 45 and 64yo
    transitions_45_64 = cp.array([[1, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0.5, 0, 0, 0, 0.5], 
                                    [0, 0, 0, 0.3, 0, 0, 0.7],
                                    [0, 0, 0, 0, 0.3, 0, 0.7],
                                    [0, 0, 0, 0, 0, 0.5, 0.5],
                                    [0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 1]])

    # For people younger between 65 and 75yo
    transitions_65_74 = cp.array([[1, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0.5, 0, 0, 0, 0.5], 
                                    [0, 0, 0, 0.3, 0, 0, 0.7],
                                    [0, 0, 0, 0, 0.3, 0, 0.7],
                                    [0, 0, 0, 0, 0, 0.5, 0.5],
                                    [0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 1]])

    # For people younger >= 75yo
    transitions_75 = cp.array([[1, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0.5, 0, 0, 0, 0.5], 
                                [0, 0, 0, 0.3, 0, 0, 0.7],
                                [0, 0, 0, 0, 0.3, 0, 0.7],
                                [0, 0, 0, 0, 0, 0.5, 0.5],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1]])

    transitions = cp.stack([transitions_15, transitions_15_44, transitions_45_64, transitions_65_74, transitions_75], axis=2)


    # Define Cells
    cell_ids = cp.arange(0, N_CELLS).astype(cp.uint32)
    attractivities = cp.random.uniform(size=N_CELLS)
    unsafeties = cp.random.uniform(size=N_CELLS)
    xcoords = cp.random.uniform(low=0, high=N_SQUARES_AXIS, size=N_CELLS).reshape(-1, 1)
    ycoords = cp.random.uniform(low=0, high=N_SQUARES_AXIS, size=N_CELLS).reshape(-1, 1)
    # Define States
    unique_state_ids = cp.arange(0, 7).astype(cp.uint32)
    unique_contagiousities = cp.array([0, .9, .8, .1, .05, 0, 0])
    unique_sensitivities = cp.array([1, 0, 0, 0, 0, 0, 0])
    unique_severities = cp.array([0, .1, .8, 1, 1, 1, 0])
    transitions = cp.stack((transitions_15, transitions_15_44, transitions_45_64, transitions_65_74, transitions_75), axis=2)
    print(f'DEBUG: (benchmark) transitions.shape: {transitions.shape} (should be (7, 7, 5))')
    # Define Agents
    agent_ids = cp.arange(0, N_AGENTS).astype(cp.uint32)
    home_cell_ids = cp.random.choice(range(N_HOME_CELLS), N_AGENTS).astype(cp.uint32)
    p_moves = draw_beta(0, 1, AVG_P_MOVE, N_AGENTS).astype(cp.float16)
    least_state_ids = cp.ones(N_AGENTS)  # least severe state is state 1 for all agents
    current_state_ids = cp.random.binomial(1, p=PROP_INFECTED_AGENTS_START, size=N_AGENTS).astype(cp.uint8)
    current_state_durations = cp.zeros(N_AGENTS)
    transitions_ids = cp.random.choice(cp.arange(0, transitions.shape[2]), size=N_AGENTS).astype(cp.uint32)
    # # State durations for each agent
    durations_healthy = durations_dead = durations_recovered = cp.ones(shape=(N_AGENTS, 1)) * -1
    durations_asymptomatic = draw_beta(1, 14, 5, N_AGENTS, True)
    durations_mild = draw_beta(5, 10, 7, N_AGENTS, True)
    durations_hospital = draw_beta(1, 8, 4, N_AGENTS, True)
    durations_reanimation = draw_beta(15, 30, 21, N_AGENTS, True)


    durations = cp.stack((durations_healthy, durations_asymptomatic, durations_mild, durations_hospital, durations_reanimation, durations_dead, durations_recovered), axis=1)

    # =========== Map =============

    map = Map(cell_ids, attractivities, unsafeties, xcoords, ycoords, unique_state_ids,
              unique_contagiousities, unique_sensitivities, unique_severities, transitions, agent_ids, home_cell_ids, p_moves, least_state_ids,
              current_state_ids, current_state_durations, durations, transitions_ids, dscale=1, current_period=0, verbose=3)


    stats = {}
    t_start = time()

    for i in range(N_PERIODS):
        print(f'starting period {i}...')
        t0 = time()
        for j in range(N_MOVES_PER_PERIOD):
            t_ = time()
            map.make_move()
            print(f'move computed in {time() - t_}s')
        map.forward_all_cells()
        states_ids, state_numbers = map.get_states_numbers()
        states_ids, state_numbers = states_ids.tolist(), state_numbers.tolist()
        stats[i] = {states_ids[k]: state_numbers[k] for k in range(len(states_ids))}
        print(f'period {i} computed in {time() - t0}s')

    print(f'duration: {time() - t_start}s')
    print(stats)
