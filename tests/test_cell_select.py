import numpy as np
from scipy.spatial.distance import cdist
from time import time


# this is to recompute each time the attractivities of cells is modified during simulation



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


def vectorized_choice(prob_matrix, axis=1):
    """ 
    selects index according to weights in `prob_matrix` rows (if `axis`==0), cols otherwise 
    see https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix
    """
    s = prob_matrix.cumsum(axis=axis)
    r = np.random.rand(prob_matrix.shape[1-axis]).reshape(2*(1-axis)-1, 2*axis - 1)
    k = (s < r).sum(axis=axis)
    k = k.astype(np.uint32)
    return k


for _ in range(100):
    n_cells, n_squares_per_side = 100, 10
    n_squares = n_squares_per_side ** 2

    cells = np.arange(0, n_cells)
    squares = np.arange(0, n_squares)

    attractivity_cells = np.random.uniform(size=n_cells).astype(np.float32)

    cells_squares = np.random.choice(squares, size=n_cells)

    squares_with_cells = np.unique(cells_squares)

    # TODO: remove upfront in calculation cells with attractivity 0 (non-relevant for moving agents )

    # inds_cells_in_square ??
    t0 = time()
    unique_square_ids, inverse, counts = np.unique(cells_squares, return_inverse=True, return_counts=True)
    # `counts`: number of cells for each square

    width_sample = np.max(counts)

    cells_squares = cells_squares[inverse]
    attractivity_cells = attractivity_cells[inverse]

    # Create a new square index and align it with the cells to get right cell <=> square mapping
    square_ids_cells = np.arange(0, counts.shape[0])
    square_ids_cells = np.repeat(square_ids_cells, counts)

    # create a sequential index dor the cells in the squares: 
    # 1, 2, 3... for the cells in the first square, then 1, 2, .. for the cells in the second square
    # Trick: 1. shift `counts` one to the right, remove last element and append 0 at the beginning:
    to_repeat = np.insert(counts, 0, 0)[:-1]
    to_repeat = np.cumsum(to_repeat)  # [0, ncells in square0, ncells in square 1, etc...]

    to_subtract = np.repeat(to_repeat, counts)  # repeat each element as many times as the corresponding square has cells
    inds_cells_in_square = np.arange(0, cells.shape[0])
    inds_cells_in_square = np.subtract(inds_cells_in_square, to_subtract)  # we have the right sequential order

    # Create `sample_arr`: one row for each square. The values first value in each row are the attractivity of its cell. Padded with 0.
    sample_arr = np.zeros((n_squares, width_sample))
    sample_arr[square_ids_cells, inds_cells_in_square] = attractivity_cells
    # Normalize the rows of `sample_arr` s.t. the rows are probability distribution
    sample_arr /= np.linalg.norm(sample_arr, ord=1, axis=1, keepdims=True).astype(np.float32)

    print(f'duration: {time() - t0}')
    print(sample_arr)
    print(np.sum(sample_arr, axis=1))


    #### Selection of squares where to move agents
    ## Create square coordinates
    x_coords = np.arange(0, n_squares_per_side) + .5
    y_coords = x_coords
    coords_squares = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1,2)
    ## Assign agents to each square
    n_agents = 2000000
    agent_ids = np.arange(0, n_agents)
    home_cells = np.zeros(shape=agent_ids.shape)
    possible_square_ids = np.unique(square_ids_cells)
    agents_squares = np.random.choice(possible_square_ids, size=n_agents)


    # compute sum attractivities in squares
    sum_attractivity_squares, unique_squares = sum_by_group(values=attractivity_cells, groups=square_ids_cells)

    # TODO: in map, place cells in squares according to their coords
    # Compute distance between cells, add .5 for average intra cell distance
    inter_square_dists = cdist(coords_squares, coords_squares, 'euclidean').astype(np.float32)
    inter_square_dists = np.add(inter_square_dists, .5)  # add .5: average distance intra square
    inter_square_dists = inter_square_dists[:,unique_squares]

    print(f'sum_attractivity_squares:\n{sum_attractivity_squares}\nshape: {sum_attractivity_squares.shape}')
    print(f'unique_squares:\n{unique_squares}\nshape: {unique_squares.shape}')
    # OK

    # Compute probability of sampling each square
    square_sampling_probas = 1 / inter_square_dists
    square_sampling_probas *= sum_attractivity_squares[None,:]  # row-wise multiplication
    square_sampling_probas /= np.linalg.norm(square_sampling_probas, ord=1, axis=1, keepdims=True)
    square_sampling_probas = square_sampling_probas.astype(np.float32)

    # select agents to move
    n_moving_agents = int(n_agents / 10)
    agents_to_move = np.random.choice(agent_ids, size=n_moving_agents)

    agents_squares_to_move = agents_squares[agents_to_move]

    # Compute number of agents by square
    unique_square_ids, inverse, counts = np.unique(agents_squares_to_move, return_inverse=True, return_counts=True)
    print(f'max unique_squares_ids: {np.max(unique_square_ids)}')

    # Select only rows corresponding to squares where there are agents to move
    square_sampling_probas = square_sampling_probas[unique_square_ids,:]

    # Apply "repeat sample" trick
    square_sampling_probas = np.repeat(square_sampling_probas, counts, axis=0)
    print(f'square_sampling_probas.shape: {square_sampling_probas.shape}')  # OK

    # Chose one square for each row (agent), considering each row as a sample proba
    squares_to_move = vectorized_choice(square_sampling_probas)
    print(f'max squares_to_move: {np.max(squares_to_move)}')

    # Now select cells in the squares where the agents move
    unique_squares_to_move, inverse, counts = np.unique(squares_to_move, return_inverse=True, return_counts=True)
    print(f'max unique_squares_to_move: {np.max(unique_squares_to_move)}')
    sample_arr_move = sample_arr[unique_squares_to_move]
    sample_arr_move = np.repeat(sample_arr_move, counts, axis=0)
    print(f'sample_arr_move.shape: {sample_arr_move.shape}')
    sample_arr_move = sample_arr_move.astype(np.float16)
    cells_to_move = vectorized_choice(sample_arr_move)
    # Now we have like "cell 2 in square 1, cell n in square 2 etc." we have to go back to the actual cell id
    print(f'squares_to_move:\n{squares_to_move}')
    to_add = to_repeat[squares_to_move]
    cells_to_move = np.add(cells_to_move, to_add)

    print(f'cells_to_move.shape[0] == cells_to_move.shape[0]? {cells_to_move.shape[0] == n_moving_agents}')
    print(f'max cells_to_move: {np.max(cells_to_move)}')



# Now proceed to contaminations in cells...