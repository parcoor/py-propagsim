from numpy import isinf, array, zeros, arange, add, subtract, cumsum, insert, unique, repeat, max, lexsort, empty
from numpy.linalg import norm
from numpy.random import rand
import numpy as np
from scipy.spatial.distance import cdist
import warnings


warnings.filterwarnings('ignore', category=RuntimeWarning) 


def get_least_severe_state(states, durations):
    """ Get the state that has the least severity > 0 """
    ids, severities = [], []
    for state in states:
        ids.append(state.get_severity())
        severities.append(state.get_severity())
    ids, severities = np.array(ids), no.array(severities)
    ind_min = np.argmin(severities)
    return states[ind_min], durations[ind_min]


def squarify(xcoords, ycoords, width_square):
    xcoords_square = (xcoords / width_square).astype(np.int32)
    ycoords_square = (ycoords / width_square).astype(np.int32)
    coords_squares = np.array(np.meshgrid(xcoords_square, ycoords_square)).T.reshape(-1,2)
    coords_squares += width_square / 2
    square_ids_cells = np.arange(0, coords_squares.shape[0])
    coords_squares = np.unique(coords_squares, axis=0)
    return coords_squares, square_ids_cells


def get_square_sampling_probas(attractivity_cells, square_ids_cells, coords_squares, intra_square_dist=.5):
    # compute sum attractivities in squares
    sum_attractivity_squares, unique_squares = sum_by_group(values=self.attractivity_cells_arr, groups=square_ids_cells)
    # Compute distances between all squares and squares having sum_attractivity > 0
    eligible_squares = unique_squares[sum_attractivity_squares > 0]
    # Compute distance between cells, add `intra_square_dist` for average intra cell distance
    inter_square_dists = cdist(coords_squares, coords_squares[eligible_squares,:], 'euclidean').astype(np.float32)
    inter_square_dists = np.add(inter_square_dists, intra_square_dist)  # add .5: average distance intra square
    inter_square_dists = inter_square_dists[:,unique_squares]
    # Compute probability of sampling each square
    square_sampling_probas = 1 / inter_square_dists
    square_sampling_probas *= sum_attractivity_squares[None,:]  # row-wise multiplication
    square_sampling_probas /= norm(square_sampling_probas, ord=1, axis=1, keepdims=True)
    square_sampling_probas = square_sampling_probas.astype(np.float32)
    return square_sampling_probas


def get_cell_sampling_probas(attractivity_cells, square_ids_cells):
    unique_square_ids, inverse, counts = np.unique(square_ids_cells, return_inverse=True, return_counts=True)
    # `counts`: number of cells for each square
    width_sample = np.max(counts)
    square_ids_cells = square_ids_cells[inverse]
    attractivity_cells = attractivity_cells_arr[inverse]
    # Create a new square index and align it with the cells to get right cell <=> square mapping
    new_square_ids = np.arange(0, counts.shape[0])
    new_square_ids = np.repeat(new_square_ids, counts)
    # create a sequential index dor the cells in the squares: 
    # 1, 2, 3... for the cells in the first square, then 1, 2, .. for the cells in the second square
    # Trick: 1. shift `counts` one to the right, remove last element and append 0 at the beginning:
    cell_index_shift = np.insert(counts, 0, 0)[:-1]
    cell_index_shift = np.cumsum(to_repeat)  # [0, ncells in square0, ncells in square 1, etc...]
    to_subtract = np.repeat(cell_index_shift, counts)  # repeat each element as many times as the corresponding square has cells
    inds_cells_in_square = np.arange(0, cells.shape[0])
    inds_cells_in_square = np.subtract(inds_cells_in_square, to_subtract)  # we have the right sequential order
    # Create `sample_arr`: one row for each square. The values first value in each row are the attractivity of its cell. Padded with 0.
    cell_sampling_probas = np.zeros((n_squares, width_sample))
    cell_sampling_probas[new_square_ids, inds_cells_in_square] = attractivity_cells
    # Normalize the rows of `sample_arr` s.t. the rows are probability distribution
    cell_sampling_probas /= np.linalg.norm(cell_sampling_probas, ord=1, axis=1, keepdims=True).astype(np.float32)
    return cell_sampling_probas, cell_index_shift


def vectorized_choice(prob_matrix, axis=1):
    """ 
    selects index according to weights in `prob_matrix` rows (if `axis`==0), cols otherwise 
    see https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix
    """
    s = prob_matrix.cumsum(axis=axis)
    r = np.random.rand(prob_matrix.shape[1-axis]).reshape(2*(1-axis)-1, 2*axis - 1)
    k = (s < r).sum(axis=axis)
    return k


def group_max(data, groups):
    order = np.lexsort((data, groups))
    groups = groups[order] # this is only needed if groups is unsorted
    data = data[order]
    index = np.empty(groups.shape[0], 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    return data[index], index


def sum_by_group(values, groups):
    """ see: https://stackoverflow.com/questions/4373631/sum-array-by-number-in-numpy 
    alternative method with meshgrid led to memory error """
    order = np.argsort(groups)
    groups = groups[order]
    values = values[order]
    values.cumsum(out=values)
    index = np.ones(groups.shape[0], 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    values = values[index]
    groups = groups[index]
    values[1:] = values[1:] - values[:-1]
    return values, groups
