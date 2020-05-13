import numpy as np
import pandas as pd
import json, os
from datetime import datetime, timedelta


"""
DURATIONS
Summary: for 89% of <70yo, 87% of 70-80yo and 82% of >80yo hospitalization lasts in average 14 days. For the rest, it lasts in average 0.67 days
Same for ICUs, with 7 days in average (source: "Estimating the burden of SARS-CoV-2 in France" Salje et al. 2020)
Asymptomatic non-contagious lasts in average 4.5 days
Asymptomatic contagious: 1 day
Infected (mild): in average 14 days
recovered contagious: 1-2 days
"""

"""
"il n'y a quasiment pas de patients qui vont directement en réanimation, fait remarquer Bernard Hoen.
 Dans tous les sites hospitaliers, on hospitalise d'abord dans une unité dédiée où les patients sont surveillés.
 (francetvinfo)
 => la modélisation colle avec la réalité
"""

states =  ['healthy', 'asymptomatic', 'asympcont', 'infected', 'hosp', 'icu', 'death', 'recovercont', 'recovered']
states2ids = {state: i for i, state in enumerate(states)}

# state_mm = {'asymptomatic': (6, 5), 'infected': (9, 7), 'hosp': (10, 4), 'icu': (18, 17), 'recovercont': (14, 12)}

DATA_DIR = os.path.join(*['..', '..', 'data'])

def get_under_params(mean, mediane):
    mu = np.log(mediane)
    sigma = np.sqrt(2 * (np.log(mean) - mu))
    return mu, sigma

def draw_lognormal(mean, mediane, n):
    mu, sigma = get_under_params(mean, mediane)
    return np.random.lognormal(mean=mu, sigma=sigma, size=n)


def split_population(pop_total):
    """ returns by age and sex the effectif corresponding to this demographic """
    pgenderpop_path = os.path.join(DATA_DIR, 'p_gender_pop.json')
    with open(pgenderpop_path, 'r') as f:
        p_gender_pop = json.load(f)
    genders, ages, effectifs  = [], [], []
    for gender, repartion in p_gender_pop.items():
        for age, prop in repartion.items():
            genders.append(gender)
            ages.append(int(age))
            effectif = round(prop * pop_total / 100)
            effectifs.append(effectif)
    res = pd.DataFrame({'gender': genders, 'agegroup': ages, 'effectif': effectifs})
    return res


def get_durations(split_pop, state_mm):
    """ state_mm: state -> (mean, median) dict for the mean and median of log distribution of the duration of each state
    possibility to refine by demography """
    res = np.ones((split_pop['effectif'].sum(), len(states))) * -1  # default duration is -1

    n = res.shape[0]

    # First we proceed the states the durations that are the same for all demographics
    for state, mm in state_mm.items():
        res[:, states2ids.get(state)] = draw_lognormal(mm[0], mm[1], n)
    # hospitalisation and ICU durations are a bit more cumbersome, see above (~15% go directly hosp -> icu)
    inds_hospicu = [4, 5]
    # < 70yo
    n = split_pop[split_pop['agegroup'] < 70]['effectif'].sum()
    inds_fast = np.random.choice(range(n), size=int(.11 * n), replace=False)
    if inds_fast.shape[0] > 0:
        res[inds_fast[:, None], inds_hospicu] = 1
    # 70-80yo
    shift = split_pop[split_pop['agegroup'] < 70]['effectif'].sum()
    n = split_pop[split_pop['agegroup'] == 70]['effectif'].values[0]
    inds_fast = np.random.choice(range(n), size=int(.13 * n), replace=False) + shift
    if inds_fast.shape[0] > 0:
        res[inds_fast[:, None], inds_hospicu] = 1
    # > 80yo
    shift = split_pop[split_pop['agegroup'] < 80]['effectif'].sum()
    n = split_pop[split_pop['agegroup'] == 80]['effectif'].values[0]
    inds_fast = np.random.choice(range(n), size=int(.18 * n), replace=False) + shift
    if inds_fast.shape[0] > 0:
        res[inds_fast[:, None], inds_hospicu] = 1
    res = np.around(res).astype(int)
    res[res==0] = 1
    return res


def get_current_state_durations(split_pop, state_mm, day):
    base_state_repartition = {}  # everybody is healthy
    for _, row in split_pop.iterrows():
        demography = f"{row['gender']}_{row['agegroup']}"
        n_state = row['effectif']
        base_state_repartition[demography] = {'state_ids': np.zeros(n_state), 'state_durations': np.ones(n_state) * -1}

    repartition_df = pd.read_csv(os.path.join(DATA_DIR, 'state_repartition_demography.csv'))
    repartition_df['day'] = pd.to_datetime(repartition_df['day'])
    day_df = repartition_df.loc[repartition_df['day'] == day].reset_index(drop=True)
    pop_totale = split_pop['effectif'].sum()
    demography_state_duration = {}
    for _, row in day_df.iterrows():
        current_state_ids, current_state_durations = np.array([]), np.array([])
        demography = row['demography']
        for state in states:
            if state not in row:
                continue
            state_id = states2ids.get(state)
            n_state = int(row[state] * pop_totale / 67000000)
            current_state_ids = np.append(current_state_ids, np.ones(n_state) * state_id)
            if state in ['recovered', 'death']:
                state_durations = np.ones(n_state) * -1
                current_state_durations = np.append(current_state_durations, state_durations)
                continue
            if state == 'asymptomatic':
                state_durations = np.ones(n_state) * 6
                current_state_durations = np.append(current_state_durations, state_durations)
                continue
            mean, mediane = state_mm.get(state)
            # TODO: check state_durations
            if state_id == 5:
                mediane = mediane -2
                state_durations = draw_lognormal(.75*mean, .75*mediane, n_state)
            else:
                state_durations = draw_lognormal(2.5*mean, 2.5*mediane, n_state)
            current_state_durations = np.append(current_state_durations, state_durations)
        current_state_durations = np.around(current_state_durations).astype(np.uint32)
        demography_state_duration[demography] = {'state_ids': current_state_ids, 'state_durations': current_state_durations}

    # Combine `base_state_repartition` and `demography_state_duration`
    current_state_ids, current_state_durations = np.array([]), np.array([])
    for demography in base_state_repartition.keys():
        states_to_insert = demography_state_duration[demography]['state_ids']
        durations_to_insert = demography_state_duration[demography]['state_durations']
        max_pos = states_to_insert.shape[0]
        base_state_repartition[demography]['state_ids'][:max_pos] = states_to_insert
        base_state_repartition[demography]['state_durations'][:max_pos] = durations_to_insert
        current_state_ids = np.append(current_state_ids, base_state_repartition[demography]['state_ids'])
        current_state_durations = np.append(current_state_durations, base_state_repartition[demography]['state_durations'])

    return current_state_ids.astype(np.uint32), current_state_durations.astype(np.uint32)


def get_transitions_ids(split_pop):
    demography_ids = np.arange(0, split_pop.shape[0])
    effectif_arr = split_pop['effectif'].values[0]
    transition_ids = np.repeat(demography_ids, effectif_arr)
    return transition_ids


def get_cell_positions(n_cells, width, density_factor=1):
    mean = (width / 2, width / 2)
    sigma_base = width / 12  # broadest sigma s.t. the map contains like 99% of the generated coordinates
    sigma = sigma_base / density_factor  # make the distribution of the cells on the map narrower if needed  
    cov = np.array([[sigma, 0], [0, sigma]])
    cell_positions = np.random.multivariate_normal(mean=mean, cov=cov, size=n_cells)
    cell_positions[cell_positions > width] = width
    cell_positions[cell_positions < 0] = 0
    return cell_positions


def get_cell_attractivities(n_home_cells, n_public_cells, avg=.5, p_closed=0):
    """
    :params n_home_cells: number of home cells (with attractivity = 0, is not mandatory but we will use it for calibration)
    :params n_public_cells: n_public_cells (with variable attractivity)
    :params avg: average attractivity of the public cells
    """
    attractivities = np.zeros(n_home_cells + n_public_cells)
    sigma = 1 / 3 * p_closed # the more public cells are closed, the narrower their attractivity deviation among them
    n_open_cells = np.around((1 - p_closed) * n_public_cells).astype(np.uint32)
    attractivities_open_cells = np.random.normal(loc=.5, scale=sigma, size=n_open_cells)
    shift = attractivities.shape[0] - n_open_cells
    attractivities[shift:] = attractivities_open_cells
    return attractivities


def get_cell_unsafeties(n_cells, n_home_cells, avg):
    a, b = 2, 2 * (1 - avg) / avg  # better
    unsafeties = np.random.beta(a, b, n_cells)
    unsafeties[:n_home_cells] = 1  # home: completely unsafe
    return unsafeties


def get_transitions(split_pop):
    transitions = []
    for _, row in split_pop.iterrows():
        demography = f"{row['gender']}_{row['agegroup']}"
        fpath = os.path.join(DATA_DIR, f'{demography}.npy')
        transition = np.load(fpath)
        transitions.append(transition)
    transitions = np.dstack(transitions)
    return transitions


def get_p_moves(n_agents, avg):
    # a, b = 3 * avg / (1 - avg), 3
    a, b = 2, 2 * (1 - avg) / avg  # better
    p_moves = np.random.beta(a, b, n_agents)
    return p_moves

def evaluate(evaluations, day, n_periods):
    df = pd.read_csv(os.path.join(DATA_DIR, 'overall_cases.csv'))
    df['day'] = pd.to_datetime(df['day'])
    day_start = day + timedelta(days=1)
    day_end = day + timedelta(days=n_periods)
    df = df.loc[(df['day'] >= day_start) & (df['day'] <= day_end)]

    res = {}
    for state in ['hosp', 'icu']:
        id_state = states2ids.get(state)
        n_sims = []
        for evaluation in evaluations:
            n_sim = evaluation[1][evaluation[0] == id_state][0] * 100
            n_sims.append(n_sim)
        n_real = df[state].values
        n_sims = np.array(n_sims)
        err_pct = np.mean(np.divide(np.abs(np.subtract(n_real, n_sims)), n_real)) * 100
        res[state] = {}
        res[state]['err'] = err_pct
        res[state]['real'] = n_real
        res[state]['estimated'] = n_sims
    return res



