from classes import State, Agent, Cell, Map
from numpy import array, min, max, sum
from utils import get_least_severe_state, get_move_proba_matrix
from itertools import chain


"""
Test1: contagions
Simulation with 2 agents in the same cell who can be in 2 distinct states: 'healthy' or 'sick'.
Those agents have distinct durations `durations_1` and `durations_2` attached to those states.
After one step, agent 1 gets sick. He contaminates agent 2. Then both get healthy again after different durations
"""

state0 = State(id=0, name='healthy', contagiousity=0, sensitivity=1, severity=0)
state1 = State(id=1, name='sick', contagiousity=1, sensitivity=0, severity=0.5)


transitions = array([[1, 0], [1, 0]])
states = [state0, state1]

least_severe_state = get_least_severe_state([state0, state1])
validated = 'OK' if least_severe_state.get_id() == state1.get_id() else 'Failed'
print(f'Check for least_severe_state: {validated}')

durations_1 = (-1, 3)  # 1st state duration: undefinite, 2nd state duration: 3 timesteps (days)
durations_2 = (-1, 2)  # 1st state duration: undefinite, 2nd state duration: 2 timesteps (days)
p_move_1 = 0.3
p_move_2 = 0.5

ind_1 = Agent(id=1, p_move=p_move_1, transitions=transitions, states=states, durations=durations_1, current_state=state0, home_cell_id=1)
ind_2 = Agent(id=2, p_move=p_move_2, transitions=transitions, states=states, durations=durations_2, current_state=state0, home_cell_id=1)

position_1 = (1, 1)
attractivity_1 = 0.5
cell_1 = Cell(id=1, position=position_1, attractivity=attractivity_1, agents=[ind_1, ind_2])

for i in range(6):
    ind_1.forward()
    ind_2.forward()
    if i == 1:
        ind_1.set_state(state1)  # `ind_1` get in state 'sick'
        cell_1.update_agent_states()  # agents in the same celle than `ind_1` get also eventually infected
    validated_1 = ('OK' if ((i == 0 or i > 3) and ind_1.get_state().get_name() == state0.get_name()) or 
                    (1 <= i <= 3 and ind_1.get_state().get_name() == state1.get_name())
                    else 'Failed')
    validated_2 = ('OK' if ((i == 0 or i > 2) and ind_2.get_state().get_name() == state0.get_name()) or 
                    (1 <= i <= 2 and ind_2.get_state().get_name() == state1.get_name())
                    else 'Failed')
    print(f'step {i}, state ind_1: {ind_1.get_state()}: {validated_1}, state ind_2: {ind_2.get_state()}: {validated_2}')


"""
Test 2: get_move_proba_matrix()
This function is critical for correctly omputing moves.
It returns a matrix where each row corresponds to a cell a each colum to an agent
Each column represents the probability repartition over the cells for a next move (if one happens).
Therefore each column should 1. contain only positive values, 2. sum up to 1 and 3. have probability 0 for the current cell of the agent
(as we assume that if an agent moves, then only to a cell different of it current one)
"""
# First we add 2 more agents and cells
durations_3 = (-1, 4)  # 1st state duration: undefinite, 2nd state duration: 3 timesteps (days)
durations_4 = (-1, 1)  # 1st state duration: undefinite, 2nd state duration: 2 timesteps (days)
p_move_3 = 0.2
p_move_4 = 0.9

ind_3 = Agent(id=3, p_move=p_move_3, transitions=transitions, states=states, durations=durations_3, current_state=state0, home_cell_id=2)
ind_4 = Agent(id=4, p_move=p_move_4, transitions=transitions, states=states, durations=durations_4, current_state=state0, home_cell_id=3)

position_2, position_3 = (1, 4), (2, 2)
cell_2 = Cell(id=2, position=position_2, attractivity=attractivity_1, agents=[ind_3])
cell_3 = Cell(id=3, position=position_3, attractivity=attractivity_1, agents=[ind_4])

cells = [cell_1, cell_2, cell_3]
agents = [ind_1, ind_2, ind_3, ind_4]
id2cell = {cell.get_id(): cell for cell in cells}
id2agents = {agent.get_id(): agent for agent in agents}

pos_agents_arr = array([id2cell.get(ind.get_home_cell_id()).get_position() for ind in agents])
pos_cells_arr = array([cell.get_position() for cell in cells])
attractivity_arr = array([cell.get_attractivity() for cell in cells])

move_proba_matrix = get_move_proba_matrix(pos_cells_arr, pos_agents_arr, attractivity_arr)

# 1. Check all values are positive
validated = 'OK' if min(move_proba_matrix) >= 0 else 'Failed'
print(f'Check for only positive values in move_proba_matrix: {validated}')

# 2. Check all columns sum up to 1
sumcols = sum(move_proba_matrix, axis=0)
validated = 'OK' if (min(sumcols) == 1 and max(sumcols) == 1) else 'Failed'
print(f'Check for all columns in move_proba_matrix summing up to 1: {validated}')

# 3. Check the current (home) cell has proba zero
cellid2ind = {cell.get_id(): i for i, cell in enumerate(cells)}
validated = 'OK'
for i, agent in enumerate(agents):
    if move_proba_matrix[cellid2ind.get(agent.get_current_cell_id()), i] != 0:
        validated = 'Failed'
        break
print(f'Check for move_proba_matrix 0 for current cells: {validated}')

"""
Test 3: Map
Create a map with the `agents` and `cells` defined above
1. Check that after a move there are no duplicated agents
2. Check that after `all_home()` the cells have the same agents
"""

map = Map(cells, agents)
repartition_0 = map.get_repartition()
# 1. Check there is no duplicate agent
map.make_move()
repartition_1 = map.get_repartition()
agent_list = list(chain.from_iterable(repartition_1.values()))
validated = 'OK' if (len(agent_list) == len(list(set(agent_list))) and len(agent_list) == len(agents)) else 'Failed'
print(f'Check for no duplicated agents after move in map: {validated}')

# 2. Check the state after calling `all_home()` is same as the initial state
map.all_home()
repartition_2 = map.get_repartition()
validated = 'OK'
for cell_id, agents_cell in repartition_0.items():
    if len(repartition_2.get(cell_id)) != len(agents_cell) or set(repartition_2.get(cell_id)) != set(agents_cell):
        validated = 'Failed'
        break
print(f'Check for return to initial state after `all_home()`: {validated}')
