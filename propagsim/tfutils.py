import tensorflow as tf
from numpy.linalg import norm
from numpy.random import rand
from scipy.spatial.distance import cdist
import warnings


warnings.filterwarnings('ignore', category=RuntimeWarning) 


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


def squarify(xcoords, ycoords, width_square):
    xcoords_square = tf.math.divide(xcoords, width_square).astype(tf.int32)
    ycoords_square = tf.math.divide(ycoords, width_square).astype(tf.int32)
    coords_squares = tf.stack((xcoords_square, ycoords_square, axis=1)).T
    coords_squares = tf.math.add(coords_squares, width_square / 2)
    coords_squares, square_ids_cells = tf.unique(coords_squares, return_inverse=True,  axis=0)
    coords_squares = tf.multiply(coords_squares, width_square)
    return coords_squares, square_ids_cells


def get_square_sampling_probas(attractivity_cells, square_ids_cells, coords_squares, intra_square_dist=.5, dscale=1):
    # compute sum attractivities in squares
    sum_attractivity_squares, unique_squares = sum_by_group(values=attractivity_cells, groups=square_ids_cells)
    # Compute distances between all squares and squares having sum_attractivity > 0
    mask_attractivity = (sum_attractivity_squares > 0)
    eligible_squares = unique_squares[mask_attractivity]
    sum_attractivity_squares = sum_attractivity_squares[mask_attractivity]
    order = tf.argsort(eligible_squares)
    eligible_squares = eligible_squares[order]
    sum_attractivity_squares = sum_attractivity_squares[order]

    # Compute distance between cells, add `intra_square_dist` for average intra cell distance
    inter_square_dists = cdist(coords_squares, coords_squares[eligible_squares,:], 'euclidean').astype(tf.float32)
    inter_square_dists = tf.multiply(inter_square_dists, dscale)
    inter_square_dists = tf.math.add(inter_square_dists, intra_square_dist)  # add .5: average distance intra square
    # Compute probability of sampling each square
    square_sampling_probas = 1 / inter_square_dists
    square_sampling_probas *= sum_attractivity_squares[None,:]  # row-wise multiplication
    square_sampling_probas /= norm(square_sampling_probas, ord=1, axis=1, keepdims=True)
    square_sampling_probas = square_sampling_probas.astype(tf.float32)
    return square_sampling_probas


def get_cell_sampling_probas(attractivity_cells, square_ids_cells):
    unique_square_ids, inverse, counts = tf.unique(square_ids_cells, return_inverse=True, return_counts=True)
    # `inverse` is an re-numering of `square_ids_cells` following its order: 3, 4, 6 => 0, 1, 2
    width_sample = tf.max(counts)
    print(f'width_sample: {width_sample}')
    # create a sequential index dor the cells in the squares: 
    # 1, 2, 3... for the cells in the first square, then 1, 2, .. for the cells in the second square
    # Trick: 1. shift `counts` one to the right, remove last element and append 0 at the beginning:
    cell_index_shift = tf.insert(counts, 0, 0)[:-1]
    cell_index_shift = tf.cumsum(cell_index_shift)  # [0, ncells in square0, ncells in square 1, etc...]
    to_subtract = tf.repeat(cell_index_shift, counts)  # repeat each element as many times as the corresponding square has cells

    inds_cells_in_square = tf.arange(0, attractivity_cells.shape[0])
    inds_cells_in_square = tf.subtract(inds_cells_in_square, to_subtract)  # we have the right sequential order

    order = tf.argsort(inverse)
    inverse = inverse[order]
    attractivity_cells = attractivity_cells[order]

    # Create `sample_arr`: one row for each square. The values first value in each row are the attractivity of its cell. Padded with 0.
    cell_sampling_probas = tf.zeros((unique_square_ids.shape[0], width_sample))
    cell_sampling_probas[inverse, inds_cells_in_square] = attractivity_cells
    # Normalize the rows of `sample_arr` s.t. the rows are probability distribution
    cell_sampling_probas /= tf.linalg.norm(cell_sampling_probas, ord=1, axis=1, keepdims=True).astype(tf.float32)
    return cell_sampling_probas, cell_index_shift



def vectorized_choice(prob_matrix, axis=1):
    """ 
    selects index according to weights in `prob_matrix` rows (if `axis`==0), cols otherwise 
    see https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix
    """
    s = prob_matrix.cumsum(axis=axis)
    r = tf.random.rand(prob_matrix.shape[1-axis]).reshape(2*(1-axis)-1, 2*axis - 1)
    k = (s < r).sum(axis=axis)
    return k


def group_max(data, groups):
    order = tf.lexsort((data, groups))
    groups = groups[order] # this is only needed if groups is unsorted
    data = data[order]
    index = tf.empty(groups.shape[0], 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    return data[index], index


def sum_by_group(values, groups):
    """ see: https://stackoverflow.com/questions/4373631/sum-array-by-number-in-numpy 
    alternative method with meshgrid led to memory error """
    order = tf.argsort(groups)
    groups = groups[order]
    values = values[order]
    values.cumsum(out=values)
    index = tf.ones(groups.shape[0], 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    values = values[index]
    groups = groups[index]
    values[1:] = values[1:] - values[:-1]
    return values, groups
