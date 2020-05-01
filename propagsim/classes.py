import numpy as np
from itertools import compress
from utils import get_least_severe_state, squarify, get_square_sampling_probas, get_cell_sampling_probas, vectorized_choice
from multiprocessing import Pool, cpu_count


class State:
    def __init__(self, id, name, contagiousity, sensitivity, severity):
        """ A state can be carried by an agent. It makes the agent accordingly contagious, 
        sensitive and in a severe state.
        """
        self.id = id
        self.name = name
        self.contagiousity = contagiousity
        self.sensitivity = sensitivity
        self.severity = severity

    def __str__(self):
        return self.name

    def get_id(self):
        return self.id

    def get_name(self):
        return self.name

    def get_contagiousity(self):
        return self.contagiousity

    def get_sensitivity(self):
        return self.sensitivity

    def get_severity(self):
        return self.severity


class Agent:
    def __init__(self, id, p_move, states, transitions, durations, current_state, home_cell_id):
        self.id = id
        self.p_move = p_move
        self.states = states
        self.transitions = transitions
        self.durations = durations
        self.current_state = current_state
        self.home_cell_id = home_cell_id
        self.least_state, self.least_state_duration = get_least_severe_state(states, durations)

    def get_id(self):
        return self.id

    def get_p_move(self):
        return self.p_move

    def set_p_move(self, p_move):
        self.p_move = p_move

    def get_states(self):
        return self.states

    def set_states(self, states):
        return self.states
    
    def get_transitions(self):
        return self.transitions

    def set_transitions(self, transitions):
        self.transitions = transitions

    def get_transitions_id(self):
        return self.transitions.get_id()

    def get_transitions_arr(self):
        return self.transitions.get_arr()

    def get_durations(self):
        return self.durations

    def set_durations(self, durations):
        self.durations = durations

    def get_current_state_id(self):
        return self.current_state.get_id()

    def get_current_state_duration(self):
        for i, state in enumerate(self.states):
            if state.get_id() == self.current_state.get_id():
                return self.durations[i]

    def set_current_state(self, current_state):
        self.current_state = current_state

    def get_home_cell_id(self):
        return self.home_cell_id

    def set_home_cell_id(self, home_cell_id):
        self.home_cell_id = home_cell_id

    def get_least_state(self):
        return self.least_state

    def get_least_state_id(self):
        return self.least_state.get_id()

    def get_least_state_duration(self):
        return self.least_state_duration

    def get_contagiousity(self):
        return self.current_state.get_contagiousity()

    def get_sensitivity(self):
        return self.current_state.get_sensitivity()

    def get_severity(self):
        return self.current_state.get_severity()




class Transitions:
    def __init__(self, id, arr):
        self.id = id
        self.arr = arr.astype(np.float32)

    def get_id(self):
        return self.id

    def get_arr(self):
        return self.arr



class Cell:
    def __init__(self, id, position, attractivity, unsafety, agents):
        """A cell is figuratively a place where several agents can be together and possibly get 
        infected from an infected agent in the cell.
        A cell has also a geographic `position` (Euclidean coordinates) and an `attractivity` influencing the 
        probability of the agents in other cells to move in this cell.
        """
        self.id = id
        self.position = position
        self.attractivity = attractivity
        self.unsafety = unsafety
        self.agents = agents
    

    def get_id(self):
        return self.id
    
    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position
    
    def get_attractivity(self):
        return self.attractivity

    def set_attractivity(self, attractivity):
        self.attractivity = attractivity
    
    def get_unsafety(self):
        return self.unsafety
    
    def set_unsafety(self, unsafety):
        self.unsafety = unsafety

    def get_agents(self):
        return self.agents

    def set_agents(self, agents):
        self.agents = agents


        
class Map:
    def __init__(self, cells, agents, width_square=1, current_period=0, verbose=0):
        """ A map contains a list of `cells`, `agents` and an implementation of the 
        way agents can move from a cell to another.
        """
        self.current_period = current_period
        self.verbose = verbose

        self.cell_ids, xcoords, ycoords, attractivities, self.unsafeties = [], [], [], [], []
        for cell in cells:
            self.cell_ids.append(cell.get_id())
            coords = cell.get_position()
            xcoords.append(coords[0])
            ycoords.append(coords[0])
            attractivities.append(cell.get_attractivity())
            self.unsafeties.append(cell.get_unsafety())

        self.cell_ids = np.array(self.cell_ids, dtype=np.uint32)
        xcoords, y_coords = np.array(xcoords, dtype=np.float32), np.array(ycoords, dtype=np.float32)
        attractivities = np.array(attractivities, dtype=np.float32)
        self.unsafeties = np.array(self.unsafeties, dtype=np.float32)
        # 
        coords_squares, square_ids_cells = squarify(xcoords, ycoords, width_square)
        self.square_sampling_probas = get_square_sampling_probas(attractivities, square_ids_cells, coords_squares, width_square/2)

        mask_eligible = np.where(attractivities > 0)[0]  # only cells with attractivity > 0 are eligible for a move
        self.eligible_cells = self.cell_ids[mask_eligible]
        self.cell_sampling_probas = get_cell_sampling_probas(attractivities[mask_eligible], square_ids_cells[mask_eligible])

        self.agent_ids = []
        self.p_moves = []
        self.severities = []
        self.sensitivities = []
        self.contagiousities = []
        # per agent
        self.least_state_ids = []  
        self.least_state_duration = []
        # followings are per least severe state
        unique_least_state_ids = []
        self.least_state_severities = []  
        self.least_state_durations = [] 
        self.least_state_contagiousities = []
        self.least_state_sensitivities = []
        self.home_cell_ids = []
        self.current_state_ids = []
        self.current_state_durations = []
        self.transitions = []
        self.transitions_ids = []
        self.durations = []
        self.n_infected_period = 0
        
        for agent in agents:
            self.agent_ids.append(agent.get_id())
            self.p_moves.append(agent.get_p_move())
            self.severities.append(agent.get_severity())
            self.sensitivities.append(agent.get_sensitivity())
            self.contagiousities.append(agent.get_contagiousity())
            least_state_id = agent.get_least_state_id()
            self.least_state_ids.append(least_state_id)
            if least_state_id not in unique_least_state_ids:
                least_state = agent.get_least_state()
                self.least_state_severities.append(least_state.get_severity()) 
                self.least_state_contagiousities = [least_state.get_contagiousity()]
                self.least_state_sensitivities = [least_state.get_sensitivity()]
                unique_least_state_ids.append(least_state_id)
            self.least_state_durations.append(agent.get_least_state_duration())
            self.home_cell_ids.append(agent.get_home_cell_id())
            self.current_state_ids.append(agent.get_current_state_id())
            self.current_state_durations.append(agent.get_current_state_duration())
            transitions_id = agent.get_transitions_id()
            if transitions_id not in self.transitions_ids:
                self.transitions_ids.append(transitions_id)
                self.transitions.append(agent.get_transitions_arr())
            self.durations.append(np.array(agent.get_durations(), dtype=np.float32))
            self.n_infected_period += int(agent.get_severity() > 0)

        self.agent_ids = np.array(self.agent_ids, dtype=np.float32)
        self.p_moves = np.array(self.p_moves, dtype=np.float32)
        self.severities = np.array(self.severities, dtype=np.float32)
        self.sensitivities = np.array(self.sensitivities, dtype=np.float32)
        self.contagiousities = np.array(self.contagiousities, dtype=np.float32)
        self.least_state_ids = np.array(self.least_state_ids, dtype=np.uint8)
        self.least_state_severities = np.array(self.least_state_severities, dtype=np.float32)
        self.least_state_contagiousities = np.array(self.least_state_contagiousities)
        self.least_state_sensitivities = np.array(self.least_state_sensitivities)
        self.least_state_durations = np.array(self.least_state_durations, dtype=np.float32)
        self.home_cell_ids = np.array(self.home_cell_ids, dtype=np.uint32)
        self.current_state_ids = np.array(self.current_state_ids, dtype=np.uint8)  # no more than 255 possible states
        self.current_state_durations = np.array(self.current_state_durations, dtype=np.float32)
        self.transitions_ids = np.array(self.transitions_ids, dtype=np.uint8)  # no more than 255 possible transitions
        # Re-order transitions by ids
        order = np.argsort(self.transitions_ids)
        self.transitions_ids = self.transitions_ids[order]
        self.transitions = np.dstack(self.transitions)
        self.transitions = self.transitions[:,:, order]
        self.durations = np.vstack(self.durations)
        # Compute probas_move for agent selection
        self.probas_move = self.p_moves * (1 - self.severities)
        # Define variable for monitoring the propagation (r factor, contagion chain)
        self.r_factors = np.array([])
        # TODO: Contagion chains
        # Define arrays for agents state transitions
        self.infecting_agents, self.infected_agents, self.infected_periods = np.array([]), np.array([]), np.array([])



    def contaminate(selected_agents, selected_cells):
        """ both arguments have same length """
        selected_contagiousities = self.contagiousities[selected_agents]
        selected_sensitivities = self.sensitivities[selected_agents]
        selected_unsafeties = self.unsafeties[selected_cells]
        order_cells = np.argsort(selected_cells, kind='heapsort')
        selected_cells = np.sort(selected_cells, kind='heapsort')
        # Sort other datas
        selected_unsafeties = selected_unsafeties[order_cells]
        selected_agents = selected_agents[order_cells]
        # Find cells where max contagiousity == 0 (no contagiousity can happen there)
        max_contagiousities, _ = group_max(data=selected_contagiousities, groups=selected_cells)
        # Find cells where max sensitivitity == 0 (no contagiousity can happen there)
        max_sensitivities, _ = group_max(data=selected_sensitivities, groups=selected_cells)
        # Combine them
        mask_zero = (np.multiply(max_contagiousities, max_sensitivities) > 0)
        _, counts = np.unique(selected_cells, return_counts=True)
        mask_zero = np.repeat(mask_zero, counts)
        # select agents being on cells with max contagiousity and max sensitivity > 0 (and their corresponding data)
        selected_agents = selected_agents[mask_zero]
        selected_contagiousities = selected_contagiousities[mask_zero]
        selected_sensitivities = selected_sensitivities[mask_zero]
        selected_cells = selected_cells[mask_zero]
        selected_unsafeties = selected_unsafeties[mask_zero]
        if self.verbose > 1:
            print(f'n selected agents after removing cells with max sensitivity or max contagiousity==0: {selected_agents.shape[0]}')
        # Find for each cell which agent has the max contagiousity inside (it will be the contaminating agent)
        max_contagiousities, mask_max_contagiousities = group_max(data=selected_contagiousities, groups=selected_cells) 
        infecting_agents = selected_agents[mask_max_contagiousities]
        selected_contagiousities = selected_contagiousities[mask_max_contagiousities]
        # Select agents that can be potentially infected ("pinfected") and corresponding variables
        pinfected_mask = (selected_sensitivities > 0)
        pinfected_agents = selected_agents[pinfected_mask]
        selected_sensitivities = selected_sensitivities[pinfected_mask]
        selected_unsafeties = selected_unsafeties[pinfected_mask]
        selected_cells = selected_cells[pinfected_mask]
        # Group `selected_cells` and expand `infecting_agents` and `selected_contagiousities` accordingly
        # There is one and only one infecting agent by pinselected_agentsfected_cell so #`counts` == #`infecting_agents`
        _, counts = np.unique(selected_cells, return_counts=True)
        infecting_agents = np.repeat(infecting_agents, counts)
        selected_contagiousities = np.repeat(selected_contagiousities, counts)
        # Compute contagions
        res = np.multiply(selected_contagiousities, selected_sensitivities)
        res = np.multiply(res, selected_unsafeties)
        draw = np.random.uniform(size=infecting_agents.shape[0])
        draw = (draw < res)
        infecting_agents = infecting_agents[draw]
        infected_agents = pinfected_agents[draw]
        self.sensitivities[infected_agents] = 0  # infeced agent can't get infected anymore
        # Append `infecting_agents` and `infected_agents` to contamination chain(s)
        self.infecting_agents = np.append(self.infecting_agents, infecting_agents)
        self.infected_agents = np.append(self.infected_agents, infected_agents)
        self.infected_periods = np.append(self.infected_periods, np.repeat(current_period, infected_agents.shape[0]))
        # recompute self.probas_move (p_move * (1 - severity))
        # Update self.current_states, self.current_durations, self.contagiousities, self.sensitivity
        new_state_ids = self.least_state_ids[infected_agents]
        self.current_state_ids[infected_agents] = new_state_ids
        self.severities = self.least_state_severities[new_state_ids]
        self.contagiousities = self.least_state_contagiousities[new_state_ids]
        self.current_state_durations[infected_agents] = 0
        self.probas_move = self.p_moves * (1 - self.severities)


    def move_agents(self, selected_agents):
        agents_squares_to_move = self.agent_squares[agents_to_move]
        # Compute number of agents by square
        unique_square_ids, inverse, counts = np.unique(agents_squares_to_move, return_inverse=True, return_counts=True)
        # Select only rows corresponding to squares where there are agents to move
        square_sampling_ps = self.square_sampling_probas[unique_square_ids,:]
        # Apply "repeat sample" trick
        square_sampling_ps = np.repeat(square_sampling_ps, counts, axis=0)
        # Chose one square for each row (agent), considering each row as a sample proba
        selected_squares = vectorized_choice(square_sampling_ps)
        # Now select cells in the squares where the agents move
        unique_selected_squares, inverse, counts = np.unique(selected_squares, return_inverse=True, return_counts=True)
        cell_sampling_ps = self.cell_sampling_probas[unique_selected_squares]
        cell_sampling_ps = np.repeat(cell_sampling_ps, counts, axis=0)
        cell_sampling_ps = cell_sampling_ps.astype(np.float16)  # float16 to avoid max memory error, precision should be enough
        selected_cells = vectorized_choice(cell_sampling_ps)
        # Now we have like "cell 2 in square 1, cell n in square 2 etc." we have to go back to the actual cell id
        index_shift = self.cell_index_shift[squares_to_move]
        selected_cells = np.add(selected_cells, index_shift)
        selected_cells = self.eligible_cells[selected_cells]
        return selected_cells



    def make_move(self):
        draw = np.random.uniform(size=self.probas_move.shape[0])
        draw = (draw < self.probas_move)
        selected_agents = self.agent_ids[draw]
        selected_cells = self.move_agents(agents_to_move)
        self.contaminate(selected_agents, selected_cells)      


    def forward_all_cells(self):
        """ move all agents in map one time step forward """
        agents_durations = self.durations[:, self.current_state_ids]
        to_transit = (self.current_state_durations == agents_durations)
        self.current_state_durations += 1
        to_transit = self.agent_ids[to_transit]
        self.transit_states(to_transit)
        # Contamination at home by end of the period
        self.contaminate(self.agent_ids, self.home_cell_ids)

    
    def transit_states(self, agent_ids_transit):
        agent_current_states = self.agent_current_state_ids[agent_ids_transit]
        agent_transitions = self.agent_transitions_ids[agent_ids_transit]
        # Reorder
        order = np.argsort(agent_transitions)
        agent_ids_transit = agent_ids_transit[order]
        agent_transitions = agent_transitions[order]
        agent_current_states = agent_current_states[order]
        # Select rows corresponding to transitions to do
        transition_states = np.vstack((agent_transitions, agent_current_states))
        unique_cols, inverse, counts = np.unique(transition_states, return_inverse=True, return_counts=True, axis=1)
        transitions = transitions[unique_cols[1,:],:,unique_cols[0,:]]
        # Repeat rows according to number of agents to draw for
        transitions = np.repeat(transitions, counts, axis=0)
        # Select new states according to transition matrix
        new_states = vectorized_choice(transitions)
        new_states = new_states[inverse]
        self.agent_current_state_ids[agent_ids_transit] = new_states


    def get_states_numbers(self):
        """ For all possible states, return the number of agents in the map in this state
        returns a numpy array consisting in 2 columns: the first is the state id and the second, 
        the number of agents currently in this state on the map """
        state_ids, n_agents = np.unique(self.current_state_ids, return_counts=True)
        return state_ids, state_numbers


    def get_r_factors(self):
        return self.r_factors
            