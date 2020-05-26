import cupy as cp
import warnings
import itertools
from time import time

warnings.filterwarnings('ignore', category=RuntimeWarning) 


def append(a, b):
    lena = a.shape[0]
    a_ = cp.zeros(lena + b.shape[0])
    a_[:lena] = a
    a_[lena:] = b
    return a_


def get_least_severe_state(states):
    """ Get the state that has the least severity > 0 """
    least_severe_state = None
    for state in states:
        if state.severity == 0:
            continue
        if least_severe_state is None:
            least_severe_state = state
        elif state.get_severity() < least_severe_state.get_severity():
            least_severe_state = state
    if least_severe_state is None:  # all states have severity 0
        least_severe_state = states[0]
    return least_severe_state


def squarify(xcoords, ycoords):
    xcoords_square = xcoords.astype(cp.uint32)
    ycoords_square = ycoords.astype(cp.uint32)
    # Trick for finding unique couples of  of (xcoords_square, ycoords_square) by comparing their sum and subtraction
    # (because cupy doesn't support unique by axis)
    xy_plus = cp.add(xcoords_square, ycoords_square)
    xy_minus = cp.subtract(xcoords_square, ycoords_square)
    # index of unique sums resp. subtractions
    _, id_plus = cp.unique(xy_plus, return_index=True)
    _, id_minus = cp.unique(xy_minus, return_index=True)

    unique_indices = append(id_plus, id_minus).astype(cp.uint32)
    unique_indices, square_ids_cells = cp.unique(unique_indices, return_inverse=True)

    xcoords_unique = xcoords_square[unique_indices]
    ycoords_unique = ycoords_square[unique_indices]

    coords_squares = cp.squeeze(cp.stack((xcoords_unique, ycoords_unique), axis=0))
    print(f'DEBUG: coords_squares.shape: {coords_squares.shape}, square_ids_cells.shape: {square_ids_cells.shape}')
    return coords_squares, square_ids_cells.astype(cp.uint32)


def get_square_sampling_probas(attractivity_cells, square_ids_cells, coords_squares, dscale=1):
    # compute sum attractivities in squares
    sum_attractivity_squares, unique_squares = sum_by_group(values=attractivity_cells, groups=square_ids_cells)
    # Compute distances between all squares and squares having sum_attractivity > 0
    mask_attractivity = (sum_attractivity_squares > 0)
    eligible_squares = unique_squares[mask_attractivity]
    print(f'DEBUG: eligible_squares.shape: {eligible_squares.shape}')
    sum_attractivity_squares = sum_attractivity_squares[mask_attractivity]

    # Compute distance between cells, add `intra_square_dist` for average intra cell distance
    print(f'DEBUG: coords_squares.shape: {coords_squares.shape}')
    inter_square_dists = cdist(coords_squares, coords_squares[:,eligible_squares])
    print(f'DEBUG: inter_square_dists.shape: {inter_square_dists.shape}')
    square_sampling_probas = cp.multiply(inter_square_dists, -1 * dscale)
    square_sampling_probas = cp.exp(square_sampling_probas)
    square_sampling_probas *= sum_attractivity_squares[None,:]  # row-wise multiplication
    square_sampling_probas /= cp.linalg.norm(square_sampling_probas, ord=1, axis=1, keepdims=True)
    square_sampling_probas = square_sampling_probas.astype(cp.float32)
    print(f'DEBUG: square_sampling_probas.shape: {square_sampling_probas.shape}')
    return square_sampling_probas


def get_cell_sampling_probas(attractivity_cells, square_ids_cells):
    seq_cell_ids = cp.arange(0, attractivity_cells.shape[0]).astype(cp.uint32)
    # Re ordering everything to have cell 0, 1, 2 of square 0, cell 0, 1 of square 1 etc.
    order = cp.lexsort(cp.stack([seq_cell_ids, square_ids_cells], axis=0))
    square_ids_cells = square_ids_cells[order]
    attractivity_cells = attractivity_cells[order]

    _, inverse, counts = cp.unique(square_ids_cells, return_inverse=True, return_counts=True)
    # `inverse` is an re-numering of `square_ids_cells` following its order: 3, 4, 6 => 0, 1, 2
    width_sample = cp.max(counts)
    seq_unique_square_ids = cp.arange(0, counts.shape[0]).astype(cp.uint32)
    seq_unique_square_ids = seq_unique_square_ids[inverse]  # now squares: 0, 0, 1, 1, 1, 2...
    # create a sequential index dor the cells in the squares: 
    # 1, 2, 3... for the cells in the first square, then 1, 2, .. for the cells in the second square
    # Trick: 1. shift `counts` one to the right, remove last element and append 0 at the beginning:
    cell_index_shift = cp.zeros(counts.shape[0] + 1)
    cell_index_shift[1:] = counts
    cell_index_shift = cp.cumsum(cell_index_shift)  # [0, ncells in square0, ncells in square 1, etc...]
    to_subtract = cell_index_shift[inverse]  # repeat each element as many times as the corresponding square has cells
    # inds_cells_in_square = inds_cells_in_square[order]
    seq_cell_ids_bis = cp.arange(0, to_subtract.shape[0])
    inds_cells_in_square = cp.subtract(seq_cell_ids_bis, to_subtract)  # we have the right sequential order

    # Create `sample_arr`: one row for each square. The values first value in each row are the attractivity of its cell. Padded with 0.
    cell_sampling_probas = cp.zeros(shape=(counts.shape[0], int(width_sample)))
    cell_sampling_probas[seq_unique_square_ids.astype(cp.uint32), inds_cells_in_square.astype(cp.uint32)] = attractivity_cells
    # Normalize the rows of `sample_arr` s.t. the rows are probability distribution
    cell_sampling_probas /= cp.linalg.norm(cell_sampling_probas, ord=1, axis=1, keepdims=True).astype(cp.float32)
    print(f'DEBUG: cell_sampling_probas:\n{cell_sampling_probas}')
    return cell_sampling_probas, cell_index_shift, order



def vectorized_choice(prob_matrix,axis=1):
    """
    selects index according to weights in `prob_matrix` rows (if `axis`==0), cols otherwise
    see https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix
    """
    # s = prob_matrix.cumsum(axis=axis)
    r = cp.random.rand(prob_matrix.shape[1-axis]).reshape(2*(1-axis)-1, 2*axis - 1)
    k = (prob_matrix < r).sum(axis=axis)
    max_choice = prob_matrix.shape[axis]
    k[k>max_choice] = max_choice
    return k


def group_max(data, groups):
    order = cp.lexsort(cp.vstack((data, groups)))
    groups = groups[order] # this is only needed if groups is unsorted
    data = data[order]
    index = cp.empty(groups.shape[0], 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    return data[index], index


def sum_by_group(values, groups):
    """ see: https://stackoverflow.com/questions/4373631/sum-array-by-number-in-numpy 
    alternative method with meshgrid led to memory error """
    order = cp.argsort(groups)
    groups = groups[order]
    values = values[order]
    values = cp.cumsum(values)
    index = cp.ones(groups.shape[0], 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    values = values[index]
    groups = groups[index]
    values[1:] = values[1:] - values[:-1]
    return values, groups


def cdist(a, b):
    dist = cp.sqrt(((b[:, None] - a[:, :, None]) ** 2).sum(0))
    return dist


def repeat(data, count):
    data, count = data.tolist(), count.tolist()
    return cp.array(list(itertools.chain(*(itertools.repeat(elem, n) for elem, n in zip(data, count)))))

def get_ind_in_arr(x, y):
    """ returns the position in x of the elements in y that are in x """
    index = cp.argsort(x)
    sorted_x = x[index]
    sorted_index = cp.searchsorted(sorted_x, y)
    yindex = cp.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y
    return yindex[~mask]
