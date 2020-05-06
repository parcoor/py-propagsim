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
    xcoords_square = xcoords.astype(cp.int32)
    ycoords_square = ycoords.astype(cp.int32) 
    # Trick for finding unique couples of  of (xcoords_square, ycoords_square) by comparing their sum and subtraction
    # (because cupy doesn't support unique by axis)
    xy_plus = cp.add(xcoords_square, ycoords_square)
    xy_minus = cp.subtract(xcoords_square, ycoords_square)
    # index of unique sums resp. subtractions
    _, id_plus = cp.unique(xy_plus, return_index=True)
    _, id_minus = cp.unique(xy_minus, return_index=True)

    unique_indices = append(id_plus, id_minus).astype(cp.uint32)
    unique_indices = cp.unique(unique_indices)

    xcoords_unique = xcoords_square[unique_indices]
    ycoords_unique = ycoords_square[unique_indices]

    square_ids_cells = cp.arange(0, xcoords_unique.shape[0])
    coords_squares = cp.vstack((xcoords_unique, ycoords_unique)).T
    return coords_squares, square_ids_cells


def get_square_sampling_probas(attractivity_cells, square_ids_cells, coords_squares, dscale=1):
    # compute sum attractivities in squares
    sum_attractivity_squares, unique_squares = sum_by_group(values=attractivity_cells, groups=square_ids_cells)
    # Compute distances between all squares and squares having sum_attractivity > 0
    mask_attractivity = (sum_attractivity_squares > 0)
    eligible_squares = unique_squares[mask_attractivity]
    sum_attractivity_squares = sum_attractivity_squares[mask_attractivity]

    # Compute distance between cells, add `intra_square_dist` for average intra cell distance
    inter_square_dists = cdist(coords_squares).astype(cp.float32)
    inter_square_dists = inter_square_dists[:,eligible_squares]
    square_sampling_probas = cp.multiply(inter_square_dists, -dscale)
    square_sampling_probas = cp.exp(square_sampling_probas)
    square_sampling_probas *= sum_attractivity_squares[None,:]  # row-wise multiplication
    square_sampling_probas /= cp.linalg.norm(square_sampling_probas, ord=1, axis=1, keepdims=True)
    square_sampling_probas = square_sampling_probas.astype(cp.float32) 
    return square_sampling_probas


def get_cell_sampling_probas(attractivity_cells, square_ids_cells):
    unique_square_ids, inverse, counts = cp.unique(square_ids_cells, return_inverse=True, return_counts=True)
    # `inverse` is an re-numering of `square_ids_cells` following its order: 3, 4, 6 => 0, 1, 2
    width_sample = int(cp.max(counts).tolist())
    # create a sequential index dor the cells in the squares: 
    # 1, 2, 3... for the cells in the first square, then 1, 2, .. for the cells in the second square
    # Trick: 1. shift `counts` one to the right, remove last element and append 0 at the beginning:

    # replace insert
    cell_index_shift = cp.zeros(shape=counts.shape)
    cell_index_shift[1:] = counts[:-1]
    cell_index_shift = cp.cumsum(cell_index_shift)  # [0, ncells in square0, ncells in square 1, etc...]
    to_subtract = repeat(cell_index_shift, counts)  # repeat each element as many times as the corresponding square has cells

    inds_cells_in_square = cp.arange(0, attractivity_cells.shape[0])
    inds_cells_in_square = cp.subtract(inds_cells_in_square, to_subtract).astype(int)  # we have the right sequential order

    order = cp.argsort(inverse)
    inverse = inverse[order].astype(int)
    attractivity_cells = attractivity_cells[order]

    # Create `sample_arr`: one row for each square. The values first value in each row are the attractivity of its cell. Padded with 0.
    print(f'DEBUG: type(unique_square_ids.shape[0]): {type(unique_square_ids.shape[0])}, type(width_sample): {type(width_sample)}')
    cell_sampling_probas = cp.zeros((unique_square_ids.shape[0], width_sample))
    cell_sampling_probas[inverse, inds_cells_in_square] = attractivity_cells
    # Normalize the rows of `sample_arr` s.t. the rows are probability distribution
    cell_sampling_probas /= cp.linalg.norm(cell_sampling_probas, ord=1, axis=1, keepdims=True).astype(cp.float32)
    return cell_sampling_probas, cell_index_shift



def vectorized_choice(prob_matrix,axis=1):
    """ 
    selects index according to weights in `prob_matrix` rows (if `axis`==0), cols otherwise 
    see https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix
    """
    # s = prob_matrix.cumsum(axis=axis)
    r = cp.random.rand(prob_matrix.shape[1-axis]).reshape(2*(1-axis)-1, 2*axis - 1)
    k = (prob_matrix < r).sum(axis=axis)
    max_choice = prob_matrix.shape[axis]
    # k[k>max_choice] = max_choice
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


# For distances:
# see: https://stackoverflow.com/questions/52030458/vectorized-spatial-distance-in-python-using-numpy

def ext_arrs(A,B, precision="float64"):
    nA,dim = A.shape
    A_ext = cp.ones((nA,dim*3),dtype=precision)
    A_ext[:,dim:2*dim] = A
    A_ext[:,2*dim:] = A**2

    nB = B.shape[0]
    B_ext = cp.ones((dim*3,nB),dtype=precision)
    B_ext[:dim] = (B**2).T
    B_ext[dim:2*dim] = -2.0*B.T
    return A_ext, B_ext

def cdist(a):
    A_ext, B_ext = ext_arrs(a,a)
    dist = A_ext.dot(B_ext)
    cp.fill_diagonal(dist,0)
    dist = cp.sqrt(dist)
    return dist


def repeat(data, count):
    data, count = data.tolist(), count.tolist()
    return cp.array(list(itertools.chain(*(itertools.repeat(elem, n) for elem, n in zip(data, count)))))
