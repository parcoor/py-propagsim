from numpy.random import choice, uniform
from numpy import array, concatenate, where
from itertools import compress
from utils import get_least_severe_state, get_move_proba_matrix
from multiprocessing import Pool, cpu_count


class State:
    def __init__(self, id, name, contagiousity, sensitivity, severity):
        """ A state can be carried by an agent. It makes the agent accordingly contagious, 
        sensitive and in a severe state.
        
        :param id: id of the state
        :type id: Integer (int)
        :param name: name of the state
        :type name: String (str)
        :param contagiousity: parameter positively correlated to the probability of 
            infecting other agents in the same cell if one agent is carrying this state
        :type contagiousity: float between 0 and 1
        :param sensitivity: parameter positively correlated to the probability of getting infected by 
            another agent if one agent carrying this state is in the same cell than another agent carrying 
            a state having contagiousity > 0
        :type sensitivity: float between 0 and 1
        :param severity: measure the severity of the state
        :type severity: float between 0 and 1
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
    def __init__(self, id, p_move, transitions, states, durations, current_state, home_cell_id):
        """ An agent can move from cell to cell and get infected by another 
        agent being simoultaneously in the same cell.
        
        :param id: id of the agent
        :type id: Integer (int)
        :param p_move: Probability for an agent to move if a move is made in the <Map> where it is
        :type p_move: float between 0 and 1
        :param transitions: Markovian transition matrix between the `states`
        :type transitions: numpy array of shape (#states, #states) with rows summing up to 1
        :param states: states in which an agent can be
        :type states: list of <State>
        :param durations: durations of the states if the agent move to them
        :type durations: iterable of length #states containing strictly positive integers
        :param current_state: state of the agent at its initialization
        :type current_state: <State>
        :param home_cell_id: if of the home cell of the agent
        :type home_cell_id: int
        """
        self.id = id
        self.p_move = p_move
        self.transitions = transitions
        self.states = states
        self.durations = durations
        self.current_state = current_state
        self.home_cell_id = home_cell_id
        self.current_cell_id = home_cell_id
        # Define dicts for `states`
        self.name2state = {state.get_name(): state for state in states}
        self.name2index = {state.get_name(): i for i, state in enumerate(states)}
        self.index2state = {i: state for i, state in enumerate(states)}
        self.state_names = {state.get_name() for state in states}
        # Define variables for `current_state`
        self.n_states = len(states)
        self.current_state_age = 0
        self.current_state_index = self.name2index.get(current_state.get_name())
        self.least_severe_state = get_least_severe_state(states)
        self.been_infected = False
        self.contaminated = []  # list of ids of agents contaminated by this agent


    def get_id(self):
        return self.id


    def get_p_move(self):
        return self.p_move


    def get_home_cell_id(self):
        return self.home_cell_id


    def get_current_cell_id(self):
        return self.current_cell_id


    def set_current_cell_id(self, cell_id):
        self.current_cell_id = cell_id


    def get_state(self):
        return self.current_state


    def get_least_severe_state(self):
        return self.least_severe_state


    def set_state(self, state):
        """ Force agent to move to given state """
        self.current_state = state
        self.current_state_age = 0
        self.current_state_index = self.name2index.get(state.get_name())


    def get_infected(self):
        """ if the agent gets infected, it jumps to the state having the lowest >0 severity if 
        its current state has severity 0, otherwise it stays to the same state, for 
        preventing absurd behavior like agent getting to a less severe state after infection """
        newly_infected = False
        if self.current_state.get_severity() < self.least_severe_state.get_severity():
            self.set_state(self.least_severe_state)
            newly_infected = not self.been_infected
            self.been_infected = True
        return newly_infected
    

    def is_infected(self):
        """ an agent is considered infected if its state has a contagiousity or a severity > 0 """
        return (self.current_state.get_contagiousity() > 0 or self.current_state.get_severity() > 0) and (self.current_state.get_severity() < 1)

    
    def has_been_infected(self):
        """ if the agent has already been affected in the past (for R coeff calculation) """
        return self.been_infected


    def is_contagious(self):
        return self.current_state.get_contagiousity() > 0


    def append_contaminated(self, agent_id):
        self.contaminated.append(agent_id)


    def get_contaminated_list(self):
        return self.contaminated

    
    def reset(self, state):
        """ reset an agent to an initial state """
        self.current_state = state
        self.contaminated = []
        self.been_infected = False


    def get_next_state(self, state):
        """ randomly sample the state following `state` according to `self.transitions` """
        probas = self.transitions[self.name2index.get(state.get_name()),:]
        index_new_state = choice(self.n_states, 1, p=probas.flatten())
        new_state = self.index2state.get(index_new_state[0])
        return new_state


    def forward(self):
        """ Go one time-step forward and update agent state accordingly: if the current state of 
        the agent reaches its end, agent moves to its next state"""
        self.current_state_age += 1
        # If current state reached age to change, move to next state
        if self.current_state_age >= self.durations[self.current_state_index]:
            new_state = self.get_next_state(self.current_state)
            self.set_state(new_state)



class Cell:
    def __init__(self, id, position, attractivity, unsafety, agents):
        """A cell is figuratively a place where several agents can be together and possibly get 
        infected from an infected agent in the cell.
        A cell has also a geographic `position` (Euclidean coordinates) and an `attractivity` influencing the 
        probability of the agents in other cells to move in this cell.
        
        :param id: id of the cell
        :type id: Integer (int)
        :param position: position of the cell
        :type position: iterable of length 2 containing non-imaginary numerical values
        :param attractivity: attractivity of the cell
        :type attractivity: positive numerical value
        :param unsafety: unsafety of the cell (positively correlated to the probability than contagions happen in the cell)
        :type unsafety: numerical value between 0 and 1
        :param agents: agents initially belonging to the cell. This cell will be their home cell
        :type agents: list of <Agent>
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

    def get_attractivity(self):
        return self.attractivity

    def get_unsafety(self):
        return self.unsafety

    def set_unsafety(self, unsafety):
        self.unsafety = unsafety

    def get_agents_id(self):
        return [agent.get_id() for agent in self.agents]

    def update_agent_states(self):
        """ update the state of the agent in the cell by activing contagion. The probability for an agent to get infected is:
        (greatest contagiousity among agents in cell) * (sensitivity of agent)
        """
        most_contagious_agent = None
        for agent in self.agents:
            if (most_contagious_agent is None or 
                    agent.get_state().get_contagiousity() > most_contagious_agent.get_state().get_contagiousity()):
                most_contagious_agent = agent
        greatest_contagiousity = most_contagious_agent.get_state().get_contagiousity()
        if greatest_contagiousity == 0: 
            return  # no update if no agent in the cell is contagious
        for agent in self.agents:  # unnecessary to parallelize, there shouldn't be too many agents in a single cell
            proba_infection = greatest_contagiousity * agent.get_state().get_sensitivity() * self.unsafety
            draw = uniform()
            if draw < proba_infection:
                newly_infected = agent.get_infected()
                if newly_infected:
                    most_contagious_agent.append_contaminated(agent.get_id())

    def add_agent(self, agent, update=True):
        """ add `agent` to the cell. `update`: if to proceed to contagion within the cell or not. """
        self.agents.append(agent)
        if update:
            self.update_agent_states()

    def remove_agent(self, agent_id):
        """ Remove agent which id (caution!) is `agent_id`. No update possible here since 
        removing an agent from a cell doesn't change the state of the remaining agents """
        self.agents = [agent for agent in self.agents if agent.get_id() != agent_id]

        
class Map:
    def __init__(self, cells, agents, possible_state_ids, verbose=0):
        """ A map contains a list of `cells`, `agents` and an implementation of the 
        way agents can move from a cell to another.
        It also contains methods to get the current repartition of agents among the cells.
        
        :param cells: cells contained in the map
        :type cells: list of <Cell>
        :param agents: agents contained in the map
        :type agents: list of <Agent>
        :param possible_state_ids: all the possible states than `agents` can have
        :type possible_state_ids: list of Integer (int)
        """
        self.cells = cells
        self.agents = agents
        self.possible_state_ids = possible_state_ids
        self.verbose = verbose
        self.n_cells = len(cells)
        self.n_agents = len(agents)
        self.n_infected_agents = len([agent for agent in agents if agent.has_been_infected()])
        self.n_contagious_agents = len([agent for agent in agents if agent.is_contagious()])
        self.r = 0  # R coeff: how many agent does an infected agent infect in average
        # Define dicts to access own cells and agents
        self.id2cell = {cell.get_id(): cell for cell in cells}
        self.id2agents = {agent.get_id(): agent for agent in agents}
        
        self.pos_agents_arr = array([self.id2cell.get(ind.get_home_cell_id()).get_position() for ind in agents])
        self.pos_cells_arr = array([cell.get_position() for cell in cells])
        self.attractivity_arr = array([cell.get_attractivity() for cell in cells])

        self.move_proba_matrix = get_move_proba_matrix(self.pos_cells_arr, self.pos_agents_arr, self.attractivity_arr)


    def get_agent(self, id):
        return self.id2agents.get(id)


    def move_agent_cell(self, agent, cell, update=True):
        """ Move `agent` to `cell`. 
        `update`: if to update its new cell state after the move """
        current_cell = self.id2cell.get(agent.get_current_cell_id())
        current_cell.remove_agent(agent.get_id())
        cell.add_agent(agent, update=update)
        agent.set_current_cell_id(cell.get_id())


    def move_single_agent(self, i):
        current_agent = self.agents[i]
        probas_new_cell = self.move_proba_matrix[:,i].flatten()
        ind_new_cell = int(choice(self.n_cells, 1, p=probas_new_cell))
        new_cell = self.cells[ind_new_cell]
        self.move_agent_cell(current_agent, new_cell)
        if self.verbose >= 2:
            print(f'INFO: agent {current_agent.get_id()} moved to cell {new_cell.get_id()}')


    def move_home(self, agent, update):
        """ Move `agent` to its home cell. 
        `update`: if to update its home cell state after the move """
        if agent.get_current_cell_id() == agent.get_home_cell_id():
            return
        cell = self.id2cell.get(agent.get_home_cell_id())
        self.move_agent_cell(agent, cell, update)


    def move_home_ind(self, i, update=True):
        self.move_home(self.agents[i], update)


    def make_move(self):
        """ Select agents to move according to their probability to move and then 
        move these to a cell according to `move_proba_matrix`. If an agent is not selected for a move, 
        it's considered to return (or stay) home during this move """
        # Select agents who make a move
        draw = uniform()
        probas_move = array([(agent.get_p_move() * (1 - agent.get_state().get_severity())) for agent in self.agents])
        inds_agents2move = where(probas_move >= draw)[0]
        if self.verbose >= 1:
            print(f'INFO: {inds_agents2move.shape[0]} selected for move')

        for i in inds_agents2move:
            self.move_single_agent(i)
        
        # Move back home agents who didn't move
        inds_agents2home = list(set(list(range(self.n_agents))) - set(inds_agents2move))
        for i in inds_agents2home:
            self.move_home_ind(i)


    def all_home(self):
        """ Move all agents to their home cell """
        for i in range(self.n_agents):
            self.move_home_ind(i)


    def forward_all_cells(self):
        """ move all agents in map one time step forward """
        for agent in self.agents:
            agent.forward()
        # Update R coeff after a global forward
        new_n_infected_agents = len([agent for agent in self.agents if agent.has_been_infected()])
        if new_n_infected_agents == 0 or self.n_contagious_agents == 0:
            self.r = 0
        else:
            self.r = (new_n_infected_agents - self.n_infected_agents) / self.n_contagious_agents
        self.n_infected_agents = new_n_infected_agents
        self.n_contagious_agents = len([agent for agent in self.agents if agent.is_contagious()])


    def get_repartition(self):
        """ returns repartition of agents by cell as dict: cell_id => [agent_id in cell] """
        return {cell.get_id(): cell.get_agents_id() for cell in self.cells}


    def get_states_numbers(self):
        """ For all possible states, return the number of agents in the map in this state """
        states_numbers = {id: 0 for id in self.possible_state_ids}
        for agent in self.agents:
            states_numbers[agent.get_state().get_id()] +=1
        return states_numbers

    def get_r(self):
        return self.r
            