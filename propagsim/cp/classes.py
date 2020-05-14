import cupy as cp
import numpy as np
import os, pickle
from time import time
from utils import get_least_severe_state, squarify, get_square_sampling_probas, get_cell_sampling_probas, vectorized_choice, group_max, append, repeat, sum_by_group


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
        self.arr = arr.astype(cp.float32)

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
    def __init__(self, cell_ids, attractivities, unsafeties, xcoords, ycoords, unique_state_ids, 
        unique_contagiousities, unique_sensitivities, unique_severities, transitions, agent_ids, home_cell_ids, p_moves, least_state_ids,
        current_state_ids, current_state_durations, durations, transitions_ids, dscale=1, current_period=0, verbose=0):
        """ A map contains a list of `cells`, `agents` and an implementation of the 
        way agents can move from a cell to another. `possible_states` must be distinct.
        We let each the possibility for each agent to have its own least severe state to make the model more flexible.
        Default parameter set to None in order to be able to create an empty map and load it from disk
        `dcale` allows to weight the importance of the distance vs. attractivity for the moves to cells

        """

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
        self.durations = cp.squeeze(durations) # 2d, one row for each agent
        self.transitions_ids = transitions_ids

        # for cells: cell_ids, attractivities, unsafeties, xcoords, ycoords
        # for states: unique_contagiousities, unique_sensitivities, unique_severities, transitions
        # for agents: home_cell_ids, p_moves, least_state_ids, current_state_ids, current_state_durations, durations (3d)
        
        # Compute inter-squares proba transition matrix
        self.coords_squares, self.square_ids_cells = squarify(xcoords, ycoords)
        self.set_attractivities(attractivities)
        
        # the first cells in parameter `cells`must be home cell, otherwise modify here
        self.agent_squares = self.square_ids_cells[self.home_cell_ids]  
        cp.cuda.Stream.null.synchronize()
        # Re-order transitions by ids
        order = cp.argsort(self.transitions_ids)
        self.transitions_ids = self.transitions_ids[order]
        self.transitions = cp.dstack(self.transitions)
        self.transitions = self.transitions[:,:, order]
        cp.cuda.Stream.null.synchronize()
        # Compute upfront cumulated sum
        self.transitions = cp.cumsum(self.transitions, axis=1)

        # Compute probas_move for agent selection
        # Define variable for monitoring the propagation (r factor, contagion chain)
        self.n_contaminated_period = 0  # number of agent contaminated during current period
        self.n_diseased_period = self.get_n_diseased()
        self.r_factors = cp.array([])
        # TODO: Contagion chains
        # Define arrays for agents state transitions
        self.infecting_agents, self.infected_agents, self.infected_periods = cp.array([]), cp.array([]), cp.array([])

        


    def contaminate(self, selected_agents, selected_cells):
        """ both arguments have same length. If an agent with sensitivity > 0 is in the same cell 
        than an agent with contagiousity > 0: possibility of contagion """
        t_start = time()
        i = 0
        t0 = time()
        selected_unsafeties = self.unsafeties[selected_cells]
        selected_agents = selected_agents.astype(cp.uint32)
        selected_states = self.current_state_ids[selected_agents]
        selected_contagiousities = self.unique_contagiousities[selected_states]
        selected_sensitivities = self.unique_sensitivities[selected_states]
        print(f'ttt first part contaminate: {time() - t0}')
        # Find cells where max contagiousity == 0 (no contagiousity can happen there)
        t0 = time()
        cont_sens = cp.multiply(selected_contagiousities, selected_sensitivities)
        print(f'ttt group max sensitivities: {time() - t0}')
        # Combine them
        if cp.max(cont_sens) == 0:
            return
        t0 = time()
        mask_zero = (cont_sens > 0)
        selected_agents = selected_agents[mask_zero]
        selected_contagiousities = selected_contagiousities[mask_zero]
        selected_sensitivities = selected_sensitivities[mask_zero]
        selected_cells = selected_cells[mask_zero]
        selected_unsafeties = selected_unsafeties[mask_zero]
        print(f'ttt mask zero all: {time() - t0}')
        
        # Compute proportion (contagious agent) / (non contagious agent) by cell
        t0 = time()
        _, n_contagious_by_cell = cp.unique(selected_cells[selected_contagiousities > 0], return_counts=True)
        _, n_non_contagious_by_cell = cp.unique(selected_cells[selected_contagiousities == 0], return_counts=True)
        print(f'ttt non contagious: {time() - t0}')
        i += 1
        t0 = time()
        p_contagious = cp.divide(n_contagious_by_cell, n_non_contagious_by_cell)

        n_selected_agents = selected_agents.shape[0]
        print(f'ttt p_contagious: {time() - t0}')
  
        if self.verbose > 1:
            print(f'{n_selected_agents} selected agents after removing cells with max sensitivity or max contagiousity==0')
        if n_selected_agents == 0:
            return
        # Find for each cell which agent has the max contagiousity inside (it will be the contaminating agent)
        t0 = time()
        max_contagiousities, mask_max_contagiousities = group_max(data=selected_contagiousities, groups=selected_cells) 
        print(f'ttt max contagious: {time() - t0}')
        t0 = time()
        infecting_agents = selected_agents[mask_max_contagiousities]
        selected_contagiousities = selected_contagiousities[mask_max_contagiousities]
        print(f'ttt mask max contagious: {time() - t0}')
        # Select agents that can be potentially infected ("pinfected") and corresponding variables
        t0 = time()
        pinfected_mask = (selected_sensitivities > 0)
        pinfected_agents = selected_agents[pinfected_mask]
        selected_sensitivities = selected_sensitivities[pinfected_mask]
        selected_unsafeties = selected_unsafeties[pinfected_mask]
        selected_cells = selected_cells[pinfected_mask]
        print(f'ttt p_infected_mask: {time() - t0}')

        # Group `selected_cells` and expand `infecting_agents` and `selected_contagiousities` accordingly
        # There is one and only one infecting agent by pinselected_agentsfected_cell so #`counts` == #`infecting_agents`
        t0 = time()
        _, inverse = cp.unique(selected_cells, return_inverse=True)
        print(f'ttt inverse select cell: {time() - t0}')
        # TODO: ACHTUNG: count repeat replace by inverse here
        t0 = time()
        infecting_agents = infecting_agents[inverse]
        selected_contagiousities = selected_contagiousities[inverse]
        p_contagious = p_contagious[inverse]
        print(f'ttt p_contagious inverse: {time() - t0}')
        # Compute contagions
        t0 = time()
        res = cp.multiply(selected_contagiousities, selected_sensitivities)
        res = cp.multiply(res, selected_unsafeties)
        print(f'ttt cp.multiply: {time() - t0}')
        # Modifiy probas contamination according to `p_contagious`
        t0 = time()
        mask_p = (p_contagious < 1)
        res[mask_p] = cp.multiply(res[mask_p], p_contagious[mask_p])
        res[~mask_p] = 1 - cp.divide(1 - res[~mask_p], p_contagious[~mask_p])
        print(f'ttt res mask p: {time() - t0}')

        t0 = time()
        draw = cp.random.uniform(size=infecting_agents.shape[0])
        draw = (draw < res)
        infecting_agents = infecting_agents[draw]
        infected_agents = pinfected_agents[draw]
        n_infected_agents = infected_agents.shape[0]
        print(f'ttt n_infected draw: {time() - t0}')
        if self.verbose > 1:
            print(f'Infecting and infected agents should be all different, are they? {((infecting_agents == infected_agents).sum() == 0)}')
            print(f'Number of infected agents: {n_infected_agents}')
        t0 = time()
        self.current_state_ids[infected_agents] = self.least_state_ids[infected_agents]
        self.current_state_durations[infected_agents] = 0
        self.n_infected_period += n_infected_agents
        self.infecting_agents = append(self.infecting_agents, infecting_agents)
        self.infected_agents = append(self.infected_agents, infected_agents)
        self.infected_periods = append(self.infected_periods, cp.multiply(cp.ones(n_infected_agents), self.current_period))
        print(f'ttt final: {time() - t0}')
        print(f'contaminate computed in {time() - t_start}')


    def move_agents(self, selected_agents):
        """ First select the square where they move and then the cell inside the square """
        t0 = time()
        selected_agents = selected_agents.astype(cp.uint32)
        agents_squares_to_move = self.agent_squares[selected_agents]

        """
        order = cp.argsort(agents_squares_to_move)
        selected_agents = selected_agents[order]
        agents_squares_to_move = agents_squares_to_move[order]
        # Compute number of agents by square
        unique_square_ids, counts = cp.unique(agents_squares_to_move, return_counts=True)
        # Select only rows corresponding to squares where there are agents to move
        square_sampling_ps = self.square_sampling_probas[unique_square_ids,:]
        # Apply "repeat sample" trick
        square_sampling_ps = cp.repeat(square_sampling_ps, counts.tolist(), axis=0)
        """
        square_sampling_ps = self.square_sampling_probas[agents_squares_to_move,:]
        # Chose one square for each row (agent), considering each row as a sample proba
        selected_squares = vectorized_choice(square_sampling_ps)
        """
        order = cp.argsort(selected_squares)
        selected_agents = selected_agents[order]
        selected_squares = selected_squares[order]
        """
        if self.verbose > 1:
            print(f'{(agents_squares_to_move != selected_squares).sum()}/{selected_agents.shape[0]} agents moving outside of their square')
        # Now select cells in the squares where the agents move
        # ACHTUNG: change unique repeat to inverse
        unique_selected_squares, inverse = cp.unique(selected_squares, return_inverse=True)
        # unique_selected_squares = unique_selected_squares.astype(cp.uint16)
        cell_sampling_ps = self.cell_sampling_probas[unique_selected_squares,:]
        cell_sampling_ps = cell_sampling_ps[inverse,:]
        """
        cell_sampling_ps = cp.repeat(cell_sampling_ps, counts.tolist(), axis=0)
        cell_sampling_ps = cell_sampling_ps.astype(cp.float16)  # float16 to avoid max memory error, precision should be enough
        """
        selected_cells = vectorized_choice(cell_sampling_ps)
        # Now we have like "cell 2 in square 1, cell n in square 2 etc." we have to go back to the actual cell id
        index_shift = self.cell_index_shift[selected_squares].astype(cp.uint32)
        selected_cells = cp.add(selected_cells, index_shift)
        # return selected_agents since it has been re-ordered
        print(f'move_agents computed in {time() - t0}')
        return selected_agents, selected_cells


    def make_move(self):
        """ determine which agents to move, then move hem and proceed to the contamination process """
        probas_move = cp.multiply(self.p_moves.flatten(),  1 - self.unique_severities[self.current_state_ids])
        draw = cp.random.uniform(size=probas_move.shape[0])
        t0 = time()
        draw = (draw < probas_move)
        print(f't draw: {time() - t0}')
        t0 = time()
        selected_agents = self.agent_ids[draw]
        print(f't selected: {time() - t0}')
        t0 = time()
        selected_agents, selected_cells = self.move_agents(selected_agents)
        print(f't move_agents(): {time() - t0}')
        if self.verbose > 1:
            print(f'{selected_agents.shape[0]} agents selected for moving')
        t0 = time()
        self.contaminate(selected_agents, selected_cells)
        print(f't contaminate(): {time() - t0}')


    def forward_all_cells(self):
        """ move all agents in map one time step forward """
        agents_durations = self.durations[cp.arange(0, self.durations.shape[0]), self.current_state_ids].flatten()
        print(f'DEBUG: agents_durations.shape: {agents_durations.shape}, self.durations.shape: {self.durations.shape}, self.current_state_ids.shape: {self.current_state_ids.shape}')
        to_transit = (self.current_state_durations == agents_durations)
        self.current_state_durations += 1
        to_transit = self.agent_ids[to_transit]
        self.transit_states(to_transit)
        # Contamination at home by end of the period
        self.contaminate(self.agent_ids, self.home_cell_ids)
        # Update r and associated variables
        r = self.n_infected_period / self.n_diseased_period if self.n_diseased_period > 0 else 0
        r = cp.array([r])
        if self.verbose > 1:
            print(f'period {self.current_period}: r={r}')
        self.r_factors = append(self.r_factors, r)
        self.n_diseased_period = self.get_n_diseased()
        self.n_infected_period = 0
        #Move one period forward
        self.current_period += 1

    
    def transit_states(self, agent_ids_transit):
        if agent_ids_transit.shape[0] == 0:
            return 
        t0 = time()
        agent_ids_transit = agent_ids_transit.astype(cp.uint32)
        agent_current_states = self.current_state_ids[agent_ids_transit]
        agent_transitions = self.transitions_ids[agent_current_states]
        # Select rows corresponding to transitions to do
        transitions = self.transitions[agent_current_states,:,agent_transitions]
        # Select new states according to transition matrix
        new_states = vectorized_choice(transitions)
        self.change_state_agents(agent_ids_transit, new_states)
        print(f'transit_states computed in {time() - t0}s')


    def get_states_numbers(self):
        """ For all possible states, return the number of agents in the map in this state
        returns a numpy array consisting in 2 columns: the first is the state id and the second, 
        the number of agents currently in this state on the map """
        state_ids, n_agents = cp.unique(self.current_state_ids, return_counts=True)
        return state_ids, n_agents


    def get_n_diseased(self):
        return ((self.unique_severities[self.current_state_ids] > 0) & (self.unique_severities[self.current_state_ids] < 1)).sum()


    def get_r_factors(self):
        return self.r_factors

    
    def get_contamination_chain(self):
        return self.infecting_agents, self.infected_agents, self.infected_periods


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
            cp.save(filepath, arr)

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
        
        self.unique_state_ids = cp.squeeze(cp.load(os.path.join(savedir, 'unique_state_ids.npy')))
        self.unique_contagiousities = cp.squeeze(cp.load(os.path.join(savedir, 'unique_contagiousities.npy')))
        self.unique_sensitivities = cp.squeeze(cp.load(os.path.join(savedir, 'unique_sensitivities.npy')))
        self.unique_severities = cp.squeeze(cp.load(os.path.join(savedir, 'unique_severities.npy')))
        self.cell_ids = cp.squeeze(cp.load(os.path.join(savedir, 'cell_ids.npy')))
        self.unsafeties = cp.squeeze(cp.load(os.path.join(savedir, 'unsafeties.npy')))
        self.square_sampling_probas = cp.squeeze(cp.load(os.path.join(savedir, 'square_sampling_probas.npy')))
        self.eligible_cells = cp.squeeze(cp.load(os.path.join(savedir, 'eligible_cells.npy')))
        self.coords_squares = cp.squeeze(cp.load(os.path.join(savedir, 'coords_squares.npy')))
        self.square_ids_cells = cp.squeeze(cp.load(os.path.join(savedir, 'square_ids_cells.npy')))
        self.cell_sampling_probas = cp.squeeze(cp.load(os.path.join(savedir, 'cell_sampling_probas.npy')))
        self.cell_index_shift = cp.squeeze(cp.load(os.path.join(savedir, 'cell_index_shift.npy')))
        self.agent_ids = cp.squeeze(cp.load(os.path.join(savedir, 'agent_ids.npy')))
        self.p_moves = cp.squeeze(cp.load(os.path.join(savedir, 'p_moves.npy')))
        self.least_state_ids = cp.squeeze(cp.load(os.path.join(savedir, 'least_state_ids.npy')))
        self.unique_state_ids = cp.squeeze(cp.load(os.path.join(savedir, 'unique_state_ids.npy')))
        self.home_cell_ids = cp.squeeze(cp.load(os.path.join(savedir, 'home_cell_ids.npy')))
        self.current_state_ids = cp.squeeze(cp.load(os.path.join(savedir, 'current_state_ids.npy')))
        self.current_state_durations = cp.squeeze(cp.load(os.path.join(savedir, 'current_state_durations.npy')))
        self.agent_squares = cp.squeeze(cp.load(os.path.join(savedir, 'agent_squares.npy')))
        self.transitions = cp.squeeze(cp.load(os.path.join(savedir, 'transitions.npy')))
        self.transitions_ids = cp.squeeze(cp.load(os.path.join(savedir, 'transitions_ids.npy')))
        self.durations = cp.squeeze(cp.load(os.path.join(savedir, 'durations.npy')))
        self.r_factors = cp.squeeze(cp.load(os.path.join(savedir, 'r_factors.npy')))
        self.infecting_agents = cp.squeeze(cp.load(os.path.join(savedir, 'infecting_agents.npy')))
        self.infected_agents = cp.squeeze(cp.load(os.path.join(savedir, 'infected_agents.npy')))
        self.infected_periods = cp.squeeze(cp.load(os.path.join(savedir, 'infected_periods.npy')))

        sdict_path = os.path.join(savedir, 'params.pkl')
        with open(sdict_path, 'rb') as f:
            sdict = pickle.load(f)

        self.current_period = sdict['current_period']
        self.verbose = sdict['verbose']
        self.dscale = sdict['dcale']
        self.n_infected_period = sdict['n_infected_period']
        self.n_diseased_period = sdict['n_diseased_period']

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
        mask_eligible = cp.where(attractivities > 0)[0]  # only cells with attractivity > 0 are eligible for a move
        self.eligible_cells = self.cell_ids[mask_eligible]
        # Compute square to cell transition matrix
        self.cell_sampling_probas, self.cell_index_shift = get_cell_sampling_probas(attractivities[mask_eligible], self.square_ids_cells[mask_eligible])
        # Compute upfront cumulated sum of sampling matrices
        self.square_sampling_probas = cp.cumsum(self.square_sampling_probas, axis=1)
        self.cell_sampling_probas = cp.cumsum(self.cell_sampling_probas, axis=1)
        
