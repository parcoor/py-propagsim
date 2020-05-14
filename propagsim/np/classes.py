import numpy as np
import os, pickle
from utils import get_least_severe_state, squarify, get_square_sampling_probas, get_cell_sampling_probas, vectorized_choice, group_max, get_ind_in_arr
from utils import get_least_severe_state, squarify, get_square_sampling_probas, get_cell_sampling_probas, vectorized_choice, group_max
from time import time


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
    def __init__(self, id, p_move, states, transitions, durations, current_state, home_cell_id, current_state_duration=0, been_infected=0):
        self.id = id
        self.p_move = p_move
        self.states = states
        self.transitions = transitions
        self.durations = durations
        self.current_state = current_state
        self.home_cell_id = home_cell_id
        self.current_state_duration = current_state_duration  # how long the agent has been in this state
        self.been_infected = been_infected
        self.least_state = get_least_severe_state(states)

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
        return self.current_state_duration

    def set_current_state(self, current_state):
        self.current_state = current_state

    def get_home_cell_id(self):
        return self.home_cell_id

    def set_home_cell_id(self, home_cell_id):
        self.home_cell_id = home_cell_id

    def get_least_state_id(self):
        return self.least_state.get_id()

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
    def __init__(self, id, position, attractivity, unsafety):
        """A cell is figuratively a place where several agents can be together and possibly get 
        infected from an infected agent in the cell.
        A cell has also a geographic `position` (Euclidean coordinates) and an `attractivity` influencing the 
        probability of the agents in other cells to move in this cell.
        """
        self.id = id
        self.position = position
        self.attractivity = attractivity
        self.unsafety = unsafety
    

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


       
class Map:
    def __init__(self, cells=None, agents=None, possible_states=None, dscale=1, current_period=0, verbose=0):
        """ A map contains a list of `cells`, `agents` and an implementation of the 
        way agents can move from a cell to another. `possible_states` must be distinct.
        We let each the possibility for each agent to have its own least severe state to make the model more flexible.
        Default parameter set to None in order to be able to create an empty map and load it from disk
        `dcale` allows to weight the importance of the distance vs. attractivity for the moves to cells
        """
        if cells is None or agents is None or possible_states is None:
            return
        self.current_period = current_period
        self.verbose = verbose
        self.dscale = dscale
        self.n_infected_period = 0
        

        self.unique_state_ids, self.unique_contagiousities, self.unique_sensitivities, self.unique_severities = [], [], [], []
        for state in possible_states:
            self.unique_state_ids.append(state.get_id())
            self.unique_contagiousities.append(state.get_contagiousity())
            self.unique_sensitivities.append(state.get_sensitivity())
            self.unique_severities.append(state.get_severity())
        self.unique_state_ids = np.array(self.unique_state_ids, dtype=np.uint8)
        self.unique_contagiousities = np.array(self.unique_contagiousities, dtype=np.float32)
        self.unique_sensitivities = np.array(self.unique_sensitivities, dtype=np.float32)
        self.unique_severities = np.array(self.unique_severities, dtype=np.float32)

        self.cell_ids, xcoords, ycoords, attractivities, self.unsafeties = [], [], [], [], []
        for cell in cells:
            self.cell_ids.append(cell.get_id())
            coords = cell.get_position()
            xcoords.append(coords[0])
            ycoords.append(coords[1])
            attractivities.append(cell.get_attractivity())
            self.unsafeties.append(cell.get_unsafety())

        self.cell_ids = np.array(self.cell_ids, dtype=np.uint32)
        xcoords, ycoords = np.array(xcoords, dtype=np.float32), np.array(ycoords, dtype=np.float32)
        attractivities = np.array(attractivities, dtype=np.float32)
        self.unsafeties = np.array(self.unsafeties, dtype=np.float32)
        # Compute inter-squares proba transition matrix
        self.coords_squares, self.square_ids_cells = squarify(xcoords, ycoords)
        self.set_attractivities(attractivities)
        # Process agent
        self.agent_ids = []
        self.p_moves = []
        self.least_state_ids = []
        self.unique_state_ids = []
        self.home_cell_ids = []
        self.current_state_ids = []
        self.current_state_durations = []
        self.transitions = []
        self.transitions_ids = []
        self.durations = []
        
        for agent in agents:
            self.agent_ids.append(agent.get_id())
            self.p_moves.append(agent.get_p_move())
            least_state_id = agent.get_least_state_id()
            self.least_state_ids.append(least_state_id)
            self.home_cell_ids.append(agent.get_home_cell_id())
            self.current_state_ids.append(agent.get_current_state_id())
            self.current_state_durations.append(agent.get_current_state_duration())
            transitions_id = agent.get_transitions_id()
            if transitions_id not in self.transitions_ids:
                self.transitions_ids.append(transitions_id)
                self.transitions.append(agent.get_transitions_arr())
            self.durations.append(np.array(agent.get_durations(), dtype=np.float32))

        self.agent_ids = np.array(self.agent_ids, dtype=np.float32)
        self.p_moves = np.array(self.p_moves, dtype=np.float32)
        self.least_state_ids = np.array(self.least_state_ids, dtype=np.uint8)
        self.home_cell_ids = np.array(self.home_cell_ids, dtype=np.uint32)
        self.current_state_ids = np.array(self.current_state_ids, dtype=np.uint8)  # no more than 255 possible states
        self.current_state_durations = np.array(self.current_state_durations, dtype=np.float32)
        self.transitions_ids = np.array(self.transitions_ids, dtype=np.uint8)  # no more than 255 possible transitions
        # the first cells in parameter `cells`must be home cell, otherwise modify here
        self.agent_squares = self.square_ids_cells[self.home_cell_ids]  
        # Re-order transitions by ids
        order = np.argsort(self.transitions_ids)
        self.transitions_ids = self.transitions_ids[order]
        self.transitions = np.dstack(self.transitions)
        self.transitions = self.transitions[:,:, order]
        # Compute upfront cumulated sum
        self.transitions = np.cumsum(self.transitions, axis=1)

        self.durations = np.vstack(self.durations)
        # Compute probas_move for agent selection
        # Define variable for monitoring the propagation (r factor, contagion chain)
        self.n_contaminated_period = 0  # number of agent contaminated during current period
        self.n_diseased_period = self.get_n_diseased()
        self.r_factors = np.array([])
        # TODO: Contagion chains
        # Define arrays for agents state transitions
        self.infecting_agents, self.infected_agents, self.infected_periods = np.array([]), np.array([]), np.array([])

        


    def contaminate(self, selected_agents, selected_cells, prop_cont_factor=10, p_mask=0, family=False):
        """ both arguments have same length. If an agent with sensitivity > 0 is in the same cell 
        than an agent with contagiousity > 0: possibility of contagion
        prop_cont_factor: influence of the proportion of contagious people in a cell on contagion risk"""

        t0 = time()
        order_cells = np.argsort(selected_cells, kind='heapsort')
        print(f'sort time (numpy): {time() - t0}')
        selected_cells = np.sort(selected_cells, kind='heapsort').astype(np.uint32)
        # Sort other datas
        selected_unsafeties = self.unsafeties[selected_cells]
        selected_agents = selected_agents[order_cells].astype(np.uint32)
        selected_states = self.current_state_ids[selected_agents]
        selected_contagiousities = self.unique_contagiousities[selected_states]
        if p_mask > 0:
            pos_contagiousities = np.where(selected_contagiousities > 0)[0]
            n_switchoff = int(pos_contagiousities.shape[0] * p_mask)
            to_switchoff = np.random.choice(pos_contagiousities, size=n_switchoff, replace=False)
            selected_contagiousities[to_switchoff] = 0

        selected_sensitivities = self.unique_sensitivities[selected_states]
        # Find cells where max contagiousity == 0 (no contagiousity can happen there)
        max_contagiousities, _ = group_max(data=selected_contagiousities, groups=selected_cells)
        if self.verbose > 1:
            print(f'{max_contagiousities[max_contagiousities > 0].shape[0]} cells with contagious agent(s)')
        # Find cells where max sensitivitity == 0 (no contagiousity can happen there)
        max_sensitivities, _ = group_max(data=selected_sensitivities, groups=selected_cells)
        # Combine them
        mask_zero = ((max_contagiousities > 0) & (max_sensitivities > 0))
        _, count = np.unique(selected_cells, return_counts=True)
        mask_zero = np.repeat(mask_zero, count)
        mask_zero = (np.multiply(max_contagiousities, max_sensitivities) > 0)
        _, counts = np.unique(selected_cells, return_counts=True)
        mask_zero = np.repeat(mask_zero, counts)
        
        # select agents being on cells with max contagiousity and max sensitivity > 0 (and their corresponding data)
        selected_agents = selected_agents[mask_zero]
        selected_contagiousities = selected_contagiousities[mask_zero]
        selected_sensitivities = selected_sensitivities[mask_zero]
        selected_cells = selected_cells[mask_zero]
        selected_unsafeties = selected_unsafeties[mask_zero]
        # Compute proportion (contagious agent) / (non contagious agent) by cell
        _, n_contagious_by_cell = np.unique(selected_cells[selected_contagiousities > 0], return_counts=True)
        _, n_non_contagious_by_cell = np.unique(selected_cells[selected_contagiousities == 0], return_counts=True)
        p_contagious = np.divide(n_contagious_by_cell, n_non_contagious_by_cell)

        n_selected_agents = selected_agents.shape[0]
        if self.verbose > 1:
            print(f'{n_selected_agents} selected agents after removing cells with max sensitivity or max contagiousity==0')
        if n_selected_agents == 0:
            return
        # Find for each cell which agent has the max contagiousity inside (it will be the contaminating agent)
        max_contagiousities, mask_max_contagiousities = group_max(data=selected_contagiousities, groups=selected_cells) 
        infecting_agents = selected_agents[mask_max_contagiousities]
        selected_contagiousities = max_contagiousities

        selected_contagiousities = selected_contagiousities[mask_max_contagiousities]
        selected_contagiousities = selected_contagiousities[mask_max_contagiousities]
        # Select agents that can be potentially infected ("pinfected") and corresponding variables
        pinfected_mask = (selected_sensitivities > 0)
        pinfected_agents = selected_agents[pinfected_mask]
        selected_sensitivities = selected_sensitivities[pinfected_mask]
        selected_unsafeties = selected_unsafeties[pinfected_mask]
        selected_cells = selected_cells[pinfected_mask]

        if self.verbose > 1:
            print(f'selected_contagiousities: {selected_contagiousities}')
            print(f'selected_sensitivities: {selected_sensitivities}')
            print(f'selected_unsafeties: {selected_unsafeties}')
        # Group `selected_cells` and expand `infecting_agents` and `selected_contagiousities` accordingly
        # There is one and only one infecting agent by pinselected_agentsfected_cell so #`counts` == #`infecting_agents`
        _, counts = np.unique(selected_cells, return_counts=True)
        infecting_agents = np.repeat(infecting_agents, counts)
        selected_contagiousities = np.repeat(selected_contagiousities, counts)
        p_contagious = np.repeat(p_contagious, counts)
        # Compute contagions
        res = np.multiply(selected_contagiousities, selected_sensitivities)
        res = np.multiply(res, selected_unsafeties)
        # Modifiy probas contamination according to `p_contagious`

        """
        mask_p = (p_contagious < 1)
        res[mask_p] = np.multiply(res[mask_p], p_contagious[mask_p])
        res[~mask_p] = 1 - np.divide(1 - res[~mask_p], p_contagious[~mask_p])
        """

        draw = np.random.uniform(size=infecting_agents.shape[0])
        if family:
            draw = np.zeros(infecting_agents.shape[0])

        mask_p = (p_contagious < 1)
        res[mask_p] = np.multiply(res[mask_p], p_contagious[mask_p])
        res[~mask_p] = 1 - np.divide(1 - res[~mask_p], p_contagious[~mask_p])

        draw = np.random.uniform(size=infecting_agents.shape[0])
        draw = (draw < res)
        infecting_agents = infecting_agents[draw]
        infected_agents = pinfected_agents[draw]
        n_infected_agents = infected_agents.shape[0]
        """
        if self.verbose > 1:
            print(f'Infecting and infected agents should be all different, are they? {((infecting_agents == infected_agents).sum() == 0)}')
            print(f'Number of infected agents: {n_infected_agents}')
        """
        # self.current_state_ids[infected_agents] = self.least_state_ids[infected_agents]
        self.current_state_ids[infected_agents] = 1

        self.current_state_ids[infected_agents] = self.least_state_ids[infected_agents]
        self.current_state_ids[infected_agents] = self.least_state_ids[infected_agents]
        self.current_state_durations[infected_agents] = 0
        self.n_infected_period += n_infected_agents
        self.infecting_agents = np.append(self.infecting_agents, infecting_agents)
        self.infected_agents = np.append(self.infected_agents, infected_agents)
        self.infected_periods = np.append(self.infected_periods, np.repeat([self.current_period], n_infected_agents))


    def set_verbose(self, verbose):
        self.verbose = verbose


    def move_agents(self, selected_agents):
        """ First select the square where they move and then the cell inside the square """
        selected_agents = selected_agents.astype(np.uint32)
        agents_squares_to_move = self.agent_squares[selected_agents]

        order = np.argsort(agents_squares_to_move)
        selected_agents = selected_agents[order]
        agents_squares_to_move = agents_squares_to_move[order]
        # Compute number of agents by square
        unique_square_ids, counts = np.unique(agents_squares_to_move, return_counts=True)
        # Select only rows corresponding to squares where there are agents to move
        square_sampling_ps = self.square_sampling_probas[unique_square_ids,:]
        # Apply "repeat sample" trick
        square_sampling_ps = np.repeat(square_sampling_ps, counts, axis=0)
        # Chose one square for each row (agent), considering each row as a sample proba
        selected_squares = vectorized_choice(square_sampling_ps)
        order = np.argsort(selected_squares)
        selected_agents = selected_agents[order]
        selected_squares = selected_squares[order]
        """
        if self.verbose > 1:
            print(f'{(agents_squares_to_move != selected_squares).sum()}/{selected_agents.shape[0]} agents moving outside of their square')
        """
        # Now select cells in the squares where the agents move
        unique_selected_squares, counts = np.unique(selected_squares, return_counts=True)
        unique_selected_squares = unique_selected_squares.astype(np.uint16)
        max_sq = self.cell_sampling_probas.shape[0] - 1
        unique_selected_squares[unique_selected_squares > max_sq] = max_sq




        cell_sampling_ps = self.cell_sampling_probas[unique_selected_squares,:]
        cell_sampling_ps = np.repeat(cell_sampling_ps, counts, axis=0)
        cell_sampling_ps = cell_sampling_ps.astype(np.float16)  # float16 to avoid max memory error, precision should be enough
        selected_cells = vectorized_choice(cell_sampling_ps)
        # Now we have like "cell 2 in square 1, cell n in square 2 etc." we have to go back to the actual cell id
        max_is = self.cell_index_shift.shape[0] - 1
        selected_squares[selected_squares > max_is] = max_is
        index_shift = self.cell_index_shift[selected_squares]
        selected_cells = np.add(selected_cells, index_shift)
        # return selected_agents since it has been re-ordered
        return selected_agents, selected_cells


    def make_move(self, prop_cont_factor=10, p_mask=0):
        """ determine which agents to move, then move hem and proceed to the contamination process """
        probas_move = np.multiply(self.p_moves.flatten(),  1 - self.unique_severities[self.current_state_ids])
        # print(f'DEBUG: mean of self.p_moves.flatten(): {np.mean(self.p_moves.flatten())}')
        # print(f'DEBUG: make_move(): probas_move.shape[0]: {probas_move.shape[0]}, mean: {np.mean(probas_move)}')


        """ determine which agents to move, then move hem and proceed to the contamination process """
        probas_move = np.multiply(self.p_moves.flatten(),  1 - self.unique_severities[self.current_state_ids])
        draw = np.random.uniform(size=probas_move.shape[0])
        draw = (draw < probas_move)
        selected_agents = self.agent_ids[draw]
        selected_agents, selected_cells = self.move_agents(selected_agents)
        if self.verbose > 1:
            print(f'{selected_agents.shape[0]} agents selected for moving in {np.unique(selected_cells).shape[0]} distinct cells')
        self.contaminate(selected_agents, selected_cells, prop_cont_factor, p_mask)



    def forward_all_cells(self, tracing_rate=0):
        """ move all agents in map one time step forward """
        agents_durations = self.durations[np.arange(0, self.durations.shape[0]),self.current_state_ids]
        to_transit = (self.current_state_durations == agents_durations)
        self.current_state_durations += 1
        to_transit = self.agent_ids[to_transit]
        new_states = self.transit_states(to_transit, tracing_rate)
        # Contamination at home by end of the period
        self.contaminate(self.agent_ids, self.home_cell_ids, family=True)
        self.transit_states(to_transit)
        # Contamination at home by end of the period
        self.contaminate(self.agent_ids, self.home_cell_ids)
        self.transit_states(to_transit)
        # Contamination at home by end of the period
        self.contaminate(self.agent_ids, self.home_cell_ids)
        # Update r and associated variables
        r = self.n_infected_period / self.n_diseased_period if self.n_diseased_period > 0 else 0
        if self.verbose > 1:
            print(f'period {self.current_period}: r={r}')
        self.r_factors = np.append(self.r_factors, r)
        self.n_diseased_period = self.get_n_diseased()
        self.n_infected_period = 0
        # tracing
        #Move one period forward
        self.current_period += 1
        return new_states

    
    def transit_states(self, agent_ids_transit, tracing_rate=0):
        #Move one period forward
        self.current_period += 1
        if agent_ids_transit.shape[0] == 0:
            return 
        agent_ids_transit = agent_ids_transit.astype(np.uint32)
        agent_current_states = self.current_state_ids[agent_ids_transit]
        agent_transitions = self.transitions_ids[agent_current_states]
        # Select rows corresponding to transitions to do
        transitions = self.transitions[agent_current_states,:,agent_transitions]
        # Select new states according to transition matrix
        new_states = vectorized_choice(transitions)
        self.change_state_agents(agent_ids_transit, new_states, tracing_rate)
        return new_states


    def get_states_numbers(self):
        """ For all possible states, return the number of agents in the map in this state
        returns a numpy array consisting in 2 columns: the first is the state id and the second, 
        the number of agents currently in this state on the map """
        state_ids, n_agents = np.unique(self.current_state_ids, return_counts=True)
        return state_ids, n_agents


    def get_n_diseased(self):
        return ((self.unique_severities[self.current_state_ids] > 0) & (self.unique_severities[self.current_state_ids] < 1)).sum()


    def get_r_factors(self):
        return self.r_factors

    
    def get_contamination_chain(self):
        return self.infecting_agents, self.infected_agents, self.infected_periods


    def change_state_agents(self, agent_ids, new_state_ids, tracing_rate=0):
        """ switch `agent_ids` to `new_state_ids` """
        self.current_state_ids[agent_ids] = new_state_ids
        self.current_state_durations[agent_ids] = 0
        # Tracing
        if tracing_rate > 0:
            new_infected_agents = agent_ids[new_state_ids == 4]
            # Index of agents that just got to state "infected" in `self.infected_agents`
            inds_nia = get_ind_in_arr(self.infecting_agents, new_infected_agents)
            infected_by_nia = self.infected_agents[inds_nia].astype(np.uint32)
            mask_traced = np.random.binomial(1, p=tracing_rate, size=infected_by_nia)
            mask_traced = (mask_traced > 0)
            traced_agents = infected_by_nia[mask_traced].astype(np.uint32)
            self.p_moves[traced_agents] = np.divide(self.p_moves[traced_agents], 5)

    def change_state_agents(self, agent_ids, new_state_ids):
        """ switch `agent_ids` to `new_state_ids` """
        self.current_state_ids[agent_ids] = new_state_ids
        self.current_state_durations[agent_ids] = 0


    ### Persistence methods

    def save(self, savedir):
        """ persist map in `savedir` """
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        # Persist arrays
        dsave = {}
        dsave['unique_state_ids'] = self.unique_state_ids, 
        dsave['unique_contagiousities'] = self.unique_contagiousities, 
        dsave['unique_sensitivities'] =  self.unique_sensitivities, 
        dsave['unique_severities'] =  self.unique_severities, 
        dsave['cell_ids'] =  self.cell_ids, 
        dsave['unsafeties'] = self.unsafeties, 
        dsave['square_sampling_probas'] =  self.square_sampling_probas, 
        dsave['eligible_cells'] =  self.eligible_cells,
        dsave['coords_squares'] =  self.coords_squares,
        dsave['square_ids_cells'] =  self.square_ids_cells,
        dsave['cell_sampling_probas'] = self.cell_sampling_probas, 
        dsave['cell_index_shift'] = self.cell_index_shift,
        dsave['agent_ids'] = self.agent_ids,
        dsave['p_moves'] = self.p_moves,
        dsave['least_state_ids'] = self.least_state_ids,
        dsave['unique_state_ids'] = self.unique_state_ids,
        dsave['home_cell_ids'] = self.home_cell_ids,
        dsave['current_state_ids'] = self.current_state_ids,
        dsave['current_state_durations'] = self.current_state_durations,
        dsave['agent_squares'] = self.agent_squares
        dsave['transitions'] = self.transitions,
        dsave['transitions_ids'] = self.transitions_ids,
        dsave['durations'] = self.durations,
        dsave['r_factors'] = self.r_factors, 
        dsave['infecting_agents'] = self.infecting_agents, 
        dsave['infected_agents'] = self.infected_agents, 
        dsave['infected_periods'] = self.infected_periods

        for fname, arr in dsave.items():
            filepath = os.path.join(savedir, f'{fname}.npy')
            np.save(filepath, arr)

        # Persist scalars and other parameters
        sdict = {}
        sdict['current_period'] = self.current_period
        sdict['verbose'] = self.verbose
        sdict['dcale'] = self.dscale
        sdict['n_infected_period'] = self.n_infected_period
        sdict['n_diseased_period'] = self.n_diseased_period

        sdict_path = os.path.join(savedir, 'params.pkl')
        with open(sdict_path, 'wb') as f:
            pickle.dump(sdict, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.verbose > 0:
            print(f'Map persisted under folder: {savedir}')


    def load(self, savedir):
        """ load map that has been persisted in `savedir` through `self.save()` """
        if not os.path.isdir(savedir):
            print(f'{savedir} is not a path')
        
        self.unique_state_ids = np.squeeze(np.load(os.path.join(savedir, 'unique_state_ids.npy')))
        self.unique_contagiousities = np.squeeze(np.load(os.path.join(savedir, 'unique_contagiousities.npy')))
        self.unique_sensitivities = np.squeeze(np.load(os.path.join(savedir, 'unique_sensitivities.npy')))
        self.unique_severities = np.squeeze(np.load(os.path.join(savedir, 'unique_severities.npy')))
        self.cell_ids = np.squeeze(np.load(os.path.join(savedir, 'cell_ids.npy')))
        self.unsafeties = np.squeeze(np.load(os.path.join(savedir, 'unsafeties.npy')))
        self.square_sampling_probas = np.squeeze(np.load(os.path.join(savedir, 'square_sampling_probas.npy')))
        self.eligible_cells = np.squeeze(np.load(os.path.join(savedir, 'eligible_cells.npy')))
        self.coords_squares = np.squeeze(np.load(os.path.join(savedir, 'coords_squares.npy')))
        self.square_ids_cells = np.squeeze(np.load(os.path.join(savedir, 'square_ids_cells.npy')))
        self.cell_sampling_probas = np.squeeze(np.load(os.path.join(savedir, 'cell_sampling_probas.npy')))
        self.cell_index_shift = np.squeeze(np.load(os.path.join(savedir, 'cell_index_shift.npy')))
        self.agent_ids = np.squeeze(np.load(os.path.join(savedir, 'agent_ids.npy')))
        self.p_moves = np.squeeze(np.load(os.path.join(savedir, 'p_moves.npy')))
        self.least_state_ids = np.squeeze(np.load(os.path.join(savedir, 'least_state_ids.npy')))
        self.unique_state_ids = np.squeeze(np.load(os.path.join(savedir, 'unique_state_ids.npy')))
        self.home_cell_ids = np.squeeze(np.load(os.path.join(savedir, 'home_cell_ids.npy')))
        self.current_state_ids = np.squeeze(np.load(os.path.join(savedir, 'current_state_ids.npy')))
        self.current_state_durations = np.squeeze(np.load(os.path.join(savedir, 'current_state_durations.npy')))
        self.agent_squares = np.squeeze(np.load(os.path.join(savedir, 'agent_squares.npy')))
        self.transitions = np.squeeze(np.load(os.path.join(savedir, 'transitions.npy')))
        self.transitions_ids = np.squeeze(np.load(os.path.join(savedir, 'transitions_ids.npy')))
        self.durations = np.squeeze(np.load(os.path.join(savedir, 'durations.npy')))
        self.r_factors = np.squeeze(np.load(os.path.join(savedir, 'r_factors.npy')))
        self.infecting_agents = np.squeeze(np.load(os.path.join(savedir, 'infecting_agents.npy')))
        self.infected_agents = np.squeeze(np.load(os.path.join(savedir, 'infected_agents.npy')))
        self.infected_periods = np.squeeze(np.load(os.path.join(savedir, 'infected_periods.npy')))

        sdict_path = os.path.join(savedir, 'params.pkl')
        with open(sdict_path, 'rb') as f:
            sdict = pickle.load(f)

        self.current_period = sdict['current_period']
        self.verbose = sdict['verbose']
        self.dscale = sdict['dcale']
        self.n_infected_period = sdict['n_infected_period']
        self.n_diseased_period = sdict['n_diseased_period']


    def from_arrays(self, cell_ids, attractivities, unsafeties, xcoords, ycoords, unique_state_ids, 
        unique_contagiousities, unique_sensitivities, unique_severities, transitions, agent_ids, home_cell_ids, p_moves, least_state_ids,
        current_state_ids, current_state_durations, durations, transitions_ids, dscale=1, current_period=0, verbose=0):
        """ to initialize a map directly from the arrays """

        self.current_period = current_period
        self.verbose = verbose
        self.dscale = dscale
        self.n_infected_period = 0
        # For cells
        self.cell_ids = cell_ids
        self.attractivities = attractivities
        self.unsafeties = unsafeties
        self.xcoords = xcoords
        self.ycoords = ycoords
        # For states
        self.unique_state_ids = unique_state_ids
        self.unique_contagiousities = unique_contagiousities
        self.unique_sensitivities = unique_sensitivities
        self.unique_severities = unique_severities
        self.transitions = transitions
        # For agents
        self.agent_ids = agent_ids
        self.home_cell_ids = home_cell_ids
        self.p_moves = p_moves
        self.least_state_ids = least_state_ids
        self.current_state_ids = current_state_ids
        self.current_state_durations = current_state_durations  # how long the agents are already in their current state
        self.durations = np.squeeze(durations) # 2d, one row for each agent
        self.current_state_durations = np.minimum(self.current_state_durations, self.durations)
        self.transitions_ids = transitions_ids

        # for cells: cell_ids, attractivities, unsafeties, xcoords, ycoords
        # for states: unique_contagiousities, unique_sensitivities, unique_severities, transitions
        # for agents: home_cell_ids, p_moves, least_state_ids, current_state_ids, current_state_durations, durations (3d)
        
        # Compute inter-squares proba transition matrix
        self.coords_squares, self.square_ids_cells = squarify(xcoords, ycoords)
        self.set_attractivities(attractivities)
        
        # the first cells in parameter `cells`must be home cell, otherwise modify here
        self.agent_squares = self.square_ids_cells[self.home_cell_ids]  
        # Re-order transitions by ids
        order = np.argsort(self.transitions_ids)
        self.transitions_ids = self.transitions_ids[order]
        self.transitions = self.transitions[:,:, transitions_ids]
        # Compute upfront cumulated sum
        self.transitions = np.cumsum(self.transitions, axis=1)

        # Compute probas_move for agent selection
        # Define variable for monitoring the propagation (r factor, contagion chain)
        self.n_contaminated_period = 0  # number of agent contaminated during current period
        self.n_diseased_period = self.get_n_diseased()
        self.r_factors = np.array([])
        # TODO: Contagion chains
        # Define arrays for agents state transitions
        self.infecting_agents, self.infected_agents, self.infected_periods = np.array([]), np.array([]), np.array([])


    # For calibration: reset parameters that can change due to public policies

    def set_p_moves(self, p_moves):
        self.p_moves = p_moves

    def set_unsafeties(self, unsafeties):
        self.unsafeties = unsafeties

    def set_attractivities(self, attractivities):
        self.square_sampling_probas = get_square_sampling_probas(attractivities, 
                                                        self.square_ids_cells, 
                                                        self.coords_squares,  
                                                        self.dscale)
        mask_eligible = np.where(attractivities > 0)[0]  # only cells with attractivity > 0 are eligible for a move
        self.eligible_cells = self.cell_ids[mask_eligible]
        # Compute square to cell transition matrix
        self.cell_sampling_probas, self.cell_index_shift = get_cell_sampling_probas(attractivities[mask_eligible], self.square_ids_cells[mask_eligible])
        # Compute upfront cumulated sum of sampling matrices
        self.square_sampling_probas = np.cumsum(self.square_sampling_probas, axis=1)
        self.cell_sampling_probas = np.cumsum(self.cell_sampling_probas, axis=1)
        

