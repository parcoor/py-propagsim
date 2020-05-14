from classes import Map
import os
import numpy as np
from pprint import pprint
from simulation import get_p_moves, draw_lognormal


# Parameters calibrated for transitions
pdict_trans = {'n_moves_per_period': 3,
         'avg_p_move': 0.0028464294165809713,
         'dscale': 1,
         'density_factor': 6.900757751853491,
         'avg_unsafety': 0.6863725519674666,
         'avg_attractivity': 0.1186002675750486,
         'p_closed': 0.8810312219717485,
         'contagiousity_infected': 0.8152275355162564,
         'contagiousity_asympcont': 0.3401245897379868,
         'contagiousity_hosp': 0,
         'contagiousity_icu': 0,
         'contagiousity_recovercont': 0.22351435663942998,
         'severity_infected': 0.5157045551759367,
         'severity_recovercont': 0.04565232118665327,
         'mean_asymptomatic_t': 4.2437772937342935,
         'mean_infected_t': 6.519007831338947,
         'mean_hosp_t': 9.69142831886173,
         'mean_icu_t': 19.516578629184988}

# Parameters calibrated for moves before lockdown
pdict_moves = {'avg_attractivity': 0.28261041168902423,
               'avg_p_move': 0.1,
               'avg_unsafety': 0.95,
               'contagiousity_asympcont': 0.0021505931743505283,
               'contagiousity_infected': 0.005239221435984414,
               'contagiousity_recovercont': 0.0015279439436449188,
               'density_factor': 8.316223286287359,
               # 'dscale': 24.298494183862765,
               'dscale': 10,
               'mean_asymptomatic_t': 4.454622632634425,
               'mean_infected_t': 6.686857145366784,
               'n_moves_per_period': 18,
               'n_squares_axis': 78,
               'prop_cont_factor': 8.01733933698665, # not used anymore
               'severity_infected': 0.85,
               'severity_recovercont': 0.19075965440378387}

pdict = {'avg_attractivity': pdict_moves['avg_attractivity'],
    'n_moves_per_period': pdict_moves['n_moves_per_period'],
    'avg_p_move': pdict_moves['avg_p_move'],
    'avg_unsafety': pdict_moves['avg_unsafety'],
    'contagiousity_asympcont': pdict_moves['contagiousity_asympcont'],
    'contagiousity_infected': pdict_moves['contagiousity_infected'],
    'contagiousity_recovercont': pdict_moves['contagiousity_recovercont'],
    'contagiousity_hosp': 0,
    'contagiousity_icu': 0,
    'density_factor': pdict_moves['density_factor'],
    'dscale': pdict_moves['dscale'],
    'mean_asymptomatic_t': pdict_moves['mean_asymptomatic_t'],
    'mean_infected_t': pdict_moves['mean_infected_t'],
    'n_squares_axis': pdict_moves['n_squares_axis'],
    'prop_cont_factor': pdict_moves['prop_cont_factor'],
    'severity_infected': pdict_moves['severity_infected'],
    'severity_recovercont': pdict_moves['severity_recovercont'],
    'mean_hosp_t': pdict_trans['mean_hosp_t'],
    'mean_icu_t': pdict_trans['mean_icu_t']
}

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



N_PERIODS = 50
n_moves_per_period = pdict['n_moves_per_period']
map_path = os.path.join('maps', 'week1')

current_state_durations = np.load(os.path.join(map_path, 'current_state_durations.npy')).flatten()
current_state_ids = np.load(os.path.join(map_path, 'current_state_ids.npy')).flatten()
inds_0 = np.where(current_state_ids==0)[0]

n_infected = 100

inds_1 = np.where(current_state_ids==1)[0]
n2switch = max(current_state_ids[current_state_ids==3].shape[0] - n_infected, 0)
if n2switch > 0:
    print(f'3 to switch: {n2switch}')
    inds2switch = np.random.choice(inds_1, size=n2switch)
    # durations2switch = draw_lognormal(pdict['mean_infected_t'], pdict['mean_infected_t'] - 1, n2switch)
    durations2switch = -1
    current_state_ids[inds2switch] = 0
    current_state_durations[inds2switch] = durations2switch
    #n2switch = max(5*300 - current_state_ids[current_state_ids==1].shape[0], 0)

n2switch = max(current_state_ids[current_state_ids==1].shape[0] - 5 * n_infected, 0)
if n2switch > 0:
    print(f'1 to switch: {n2switch}')
    durations2switch = -1
    # durations2switch = draw_lognormal(pdict['mean_asymptomatic_t'], pdict['mean_asymptomatic_t'] - 1, n2switch)
    current_state_ids[inds2switch] = 0
    current_state_durations[inds2switch] = durations2switch
    np.save(os.path.join(map_path, 'current_state_durations.npy'), current_state_durations)
    np.save(os.path.join(map_path, 'current_state_ids.npy'), current_state_ids)


# agents move now `f_unmove`x less than before lockdown
f_unsafety = .33
unsafeties = np.load(os.path.join(map_path, 'unsafeties.npy')).flatten()
unsafeties = np.multiply(unsafeties, f_unsafety)


# agents move now `f_unmove`x less than before lockdown
f_unmove = 1
p_moves = np.load(os.path.join(map_path, 'p_moves.npy')).flatten()
n_p_moves = p_moves.shape[0]
p_move = pdict['avg_p_move'] / f_unmove
p_moves = get_p_moves(n_p_moves, p_move)
print(f'DEBUG: mean of p_move: {np.mean(p_moves)}')

res = {}
map = Map()
map.load(map_path)
# map.set_verbose(3)

map.set_p_moves(p_moves)
map.set_unsafeties(unsafeties)


new_hosps = []
for i in range(N_PERIODS):
    res[i] = {}
    for _ in range(n_moves_per_period):
        map.make_move(p_mask=.3)
    new_states = map.forward_all_cells(tracing_rate=0)
    new_hosp = new_states[new_states == 4].shape[0]
    new_hosps.append(new_hosp)
    state_ids, state_numbers = map.get_states_numbers()
    for j in range(state_ids.shape[0]):
        res[i][state_ids[j]] = state_numbers[j]

pprint(res)
print(new_hosps)


