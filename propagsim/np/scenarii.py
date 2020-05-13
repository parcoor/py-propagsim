from classes import Map
import os
import numpy as np
from pprint import pprint
from simulation import get_p_moves

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
n_moves_per_period = 8
map_path = os.path.join('maps', 'week1')

home_cell_ids = np.load(os.path.join(map_path, 'home_cell_ids.npy')).flatten()
square_ids_cells = np.load(os.path.join(map_path, 'square_ids_cells.npy')).flatten()

n_home_cells = np.unique(home_cell_ids).shape[0]
n_cells = square_ids_cells.shape[0]

n_public_cells = n_cells - n_home_cells

attractivities = np.zeros(n_cells)
n_open_cells = int(n_public_cells / 2)
attractivities[n_home_cells: n_home_cells+n_open_cells] = np.random.uniform(size=n_open_cells)

unsafeties = np.ones(n_cells)
# unsafeties[n_home_cells: n_home_cells+n_open_cells] = np.random.uniform(size=n_open_cells)
unsafeties[n_home_cells: n_home_cells+n_open_cells] = draw_beta(0, 1, .5, n_open_cells).flatten()


p_moves = np.load(os.path.join(map_path, 'p_moves.npy')).flatten()
n_p_moves = p_moves.shape[0]

avg_moves_period = 3
avg = avg_moves_period / n_moves_per_period
p_moves = get_p_moves(n_p_moves, avg)

res = {}
map = Map()
map.load(map_path)
map.set_attractivities(attractivities)
map.set_unsafeties(unsafeties)
map.set_p_moves(p_moves)

for i in range(N_PERIODS):
    res[i] = {}
    for _ in range(n_moves_per_period):
        map.make_move(p_mask=.2)
    map.forward_all_cells()
    state_ids, state_numbers = map.get_states_numbers()
    for j in range(state_ids.shape[0]):
        res[i][state_ids[j]] = state_numbers[j]

pprint(res)


