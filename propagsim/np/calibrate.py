from classes import State, Agent, Cell, Transitions, Map
from simulation import split_population, get_durations, get_current_state_durations, get_transitions_ids, evaluate
from simulation import get_cell_positions, get_cell_attractivities, get_cell_unsafeties, get_transitions, get_p_moves
import numpy as np
from time import time
import os, json
from datetime import datetime

CALIBRATION_DIR = os.path.join(*['..', '..', 'calibrations'])
if not os.path.isdir(CALIBRATION_DIR):
    os.makedirs(CALIBRATION_DIR)

# FIXED
DAY = datetime(2020, 4, 20)
N_PERIODS = 17
N_AGENTS = 700000
N_SQUARES_AXIS = 125
AVG_AGENTS_HOME = 2.2
N_HOME_CELLS = int(N_AGENTS / AVG_AGENTS_HOME)
PROP_PUBLIC_CELLS = 1 / 70  # there is one public place for 70 people in France
N_CELLS = int(N_HOME_CELLS + N_AGENTS * PROP_PUBLIC_CELLS)
split_pop = split_population(N_AGENTS)
states =  ['healthy', 'asymptomatic', 'asympcont', 'infected', 'hosp', 'icu', 'death', 'recovercont', 'recovered']
states2ids = {state: i for i, state in enumerate(states)}
ids2states = {v: k for k, v in states2ids.items()}



def get_random_parameters():
    pdict = {}
    pdict['n_moves_per_period'] = np.random.choice(np.arange(2, 16)).astype(np.uint16)
    avg_agent_move = np.random.uniform(.5, 2)
    pdict['avg_p_move'] = avg_agent_move / pdict['n_moves_per_period']
    pdict['dscale'] = np.random.uniform(1, 50)
    pdict['density_factor'] = np.random.uniform(1, 10)
    pdict['avg_unsafety'] = np.random.uniform(0, 1)
    pdict['avg_attractivity'] = np.random.uniform(0, 1)
    pdict['p_closed'] = np.random.uniform(.8, 1)
    pdict['contagiousity_infected'] = np.random.uniform(.3, 1)
    pdict['contagiousity_asympcont'] = np.random.uniform(0, pdict['contagiousity_infected'])
    pdict['contagiousity_hosp'] = 0
    pdict['contagiousity_icu'] = 0
    pdict['contagiousity_recovercont'] = np.random.uniform(0, pdict['contagiousity_infected'])
    pdict['severity_infected'] = np.random.uniform(.5, 1)
    pdict['severity_recovercont'] = np.random.uniform(0, pdict['severity_infected'])
    pdict['mean_asymptomatic_t'] = np.random.uniform(5, 7)
    pdict['mean_infected_t'] = np.random.uniform(4, 8)
    pdict['mean_hosp_t'] = np.random.uniform(4, 10)
    pdict['mean_icu_t'] = np.random.uniform(15, 20)
    return pdict





def build_parameters(current_period=0, verbose=0):
    pdict = get_random_parameters()

    state_mm = {'asymptomatic': (pdict['mean_asymptomatic_t'], pdict['mean_asymptomatic_t'] - 1),
                'infected': (pdict['mean_infected_t'], pdict['mean_infected_t'] - 2),
                'asympcont': (1, 1),
                'hosp': (pdict['mean_hosp_t'], pdict['mean_hosp_t'] - 2),
                'icu': (pdict['mean_icu_t'], pdict['mean_icu_t'] - 2),
                'recovercont': (2, 1)}

    # vectors
    cell_ids = np.arange(0, N_CELLS).astype(np.uint32)
    attractivities = get_cell_attractivities(N_HOME_CELLS, N_CELLS - N_HOME_CELLS, avg=pdict['avg_attractivity'], p_closed=pdict['p_closed'])
    unsafeties = get_cell_unsafeties(N_CELLS, N_HOME_CELLS, pdict['avg_unsafety'])
    cell_positions = get_cell_positions(N_CELLS, N_SQUARES_AXIS, pdict['density_factor'])
    xcoords = cell_positions[:,0]
    ycoords = cell_positions[:,1]
    unique_state_ids = np.arange(0, len(states)).astype(np.uint32)
    unique_contagiousities = np.array([0, 0, pdict['contagiousity_asympcont'], pdict['contagiousity_infected'], pdict['contagiousity_hosp'], pdict['contagiousity_icu'], 0, pdict['contagiousity_recovercont'], 0])
    unique_sensitivities = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    unique_severities = np.array([0, 0, 0, pdict['severity_infected'], 1, 1, 1, pdict['severity_recovercont'], 0])
    transitions = get_transitions(split_pop)
    current_state_ids, current_state_durations = get_current_state_durations(split_pop, state_mm, DAY)
    n_agents_generated = current_state_ids.shape[0]
    agent_ids = np.arange(0, n_agents_generated).astype(np.uint32)
    home_cell_ids = np.random.choice(np.arange(0, N_HOME_CELLS), size=n_agents_generated).astype(np.uint32)
    p_moves = get_p_moves(n_agents_generated, pdict['avg_p_move'])
    least_state_ids = np.ones(n_agents_generated)  # least severe state is state 1 for all agents

    durations = get_durations(split_pop, state_mm)
    transitions_ids = get_transitions_ids(split_pop)
    dscale = pdict['dscale']


    array_params = {'cell_ids': cell_ids, 'attractivities': attractivities, 'unsafeties': unsafeties,
                    'xcoords': xcoords, 'ycoords': ycoords, 'unique_state_ids': unique_state_ids,
                    'unique_contagiousities': unique_contagiousities, 'unique_sensitivities': unique_sensitivities,
                    'unique_severities': unique_severities, 'transitions': transitions, 'agent_ids': agent_ids,
                    'home_cell_ids': home_cell_ids, 'p_moves': p_moves, 'least_state_ids': least_state_ids,
                    'current_state_ids': current_state_ids, 'current_state_durations': current_state_durations,
                    'durations': durations, 'transitions_ids': transitions_ids, 'dscale': dscale,
                    'current_period': current_period, 'verbose': verbose}

    return array_params, pdict


def run_calibration(n_rounds, current_period=0, verbose=0):
    if not os.path.isdir('../calibrations'):
        os.makedirs('../calibrations')
    map = Map()
    best_score = None
    for i in range(n_rounds):
        if i%10 == 0:
            print(f'round {i}...')
        evaluations = []
        array_params, pdict = build_parameters(current_period, verbose)
        map.from_arrays(**array_params)
        for _ in range(N_PERIODS):
            for _ in range(pdict['n_moves_per_period']):
                map.make_move()
            map.forward_all_cells()
            state_ids, state_numbers = map.get_states_numbers()
            evaluations.append((state_ids, state_numbers))
        score = evaluate(evaluations, DAY, N_PERIODS)


        if best_score is None or (score['hosp']['err'] <= best_score['hosp']['err'] and score['icu']['err'] <= best_score['icu']['err']):
            fpath = os.path.join(CALIBRATION_DIR, f'{i}.npy')
            to_save = {'score': score, 'params': array_params, 'rt': map.get_r_factors()}
            print(f'New best score found: {score}, saved under {fpath}')
            print(f"n_moves_per_period: {pdict['n_moves_per_period']}")
            best_score = score
            np.save(fpath, to_save)
            map.save(os.path.join('maps', 'week1'))


run_calibration(n_rounds=1000)