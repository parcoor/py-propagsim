import numpy as np
from time import time


def group_max(data, groups):
    order = np.lexsort((data, groups))
    groups = groups[order] #this is only needed if groups is unsorted
    data = data[order]
    index = np.empty(groups.shape[0], 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    return data[index], index

### Setup
n_agents = 1000000
n_cells = int(0.1 * n_agents)
p_select_agent = 0.5

agent_ids, cell_ids = np.arange(0, n_agents), np.arange(0, n_cells)
unsafety_cells = np.random.uniform(size=n_cells)
selected_contagiousities = np.random.uniform(size=n_agents)
selected_sensitivities = np.random.uniform(size=n_agents)
filter_contagiousity = np.random.binomial(1, size=n_agents, p=.01)
filter_sensitivity = 1 - filter_contagiousity
selected_contagiousities = np.multiply(selected_contagiousities, filter_contagiousity)
selected_sensitivities = np.multiply(selected_sensitivities, filter_sensitivity)

selected_agents_mask = (np.random.binomial(1, p=p_select_agent, size=n_agents) > 0)
selected_agents = agent_ids[selected_agents_mask]
selected_contagiousities = selected_contagiousities[selected_agents_mask]
selected_sensitivities = selected_sensitivities[selected_agents_mask]
print(f'cell_ids.shape[0] = {cell_ids.shape[0]}')
selected_cells = np.random.choice(cell_ids, size=selected_agents.shape[0])
selected_unsafeties = unsafety_cells[selected_cells]

# take home cell ids of the agents for contamination at home of agent at the end of each period
# take agents who moved and the corresponding cells for contamination after move

################################################
t0 = time()
### Actual computation of contamination
## Sort cell ids
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
# Once contaminated, agents have sensitivity 0
# self.selected_sensitivities[infected_agents] = 0

# Append `infecting_agents` and `infected_agents` to contamination chain(s)

print(f'duration (refactor): {time() - t0}')

# check that all values are different in `infecting_agents` and `infected_agents`
print(f'Infecting and infected agents should be all different, are they? {((infecting_agents == infected_agents).sum() == 0)}')
print(f'Number of infected agents: {infected_agents.shape[0]}')
