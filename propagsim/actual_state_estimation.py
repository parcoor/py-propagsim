import requests, os
import pandas as pd
import numpy as np


states = ['hosp', 'icu', 'recovered', 'death']
path = os.path.join(*['..', 'data'])
shift = 7
shift_asymp = 5
smoothing = 7
p_asympinf = .1

### 1. Download dataframes
url_gender_fr = 'https://www.data.gouv.fr/en/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7'
url_age_fr = 'https://www.data.gouv.fr/en/datasets/r/08c18e08-6780-452d-9b8c-ae244ad529b3'
url_new_fr = 'https://www.data.gouv.fr/en/datasets/r/6fadff46-9efd-4c53-942a-54aca783c30c'

df_dict = {'gender': url_gender_fr, 'age': url_age_fr, 'new': url_new_fr}


def download_dataframes(df_dict, path):
    """
    Downloaf hospitalisaton dataframes and stores them
    :param df_dict: name -> url dict corresponding to the dataframes to download
    :param path: path to the directory where to download the dataframes
    """
    for name, url in df_dict.items():
        r = requests.get(url)
        with open(os.path.join(path, f'{name}.csv'), 'wb') as f:
            f.write(r.content)


### 2. Process genders
gender_df = pd.read_csv(os.path.join(path, 'gender.csv'), sep=';')
gender_df = gender_df.rename(columns={'sexe': 'gender', 'jour': 'day', 'rea': 'icu', 'rad': 'recovered', 'dc': 'death'})
gender_df = gender_df.groupby(['gender', 'day'])[['hosp', 'icu', 'recovered', 'death']].agg('sum').reset_index()
# Split by gender
gender_df_h = gender_df.loc[gender_df['gender'] == 1, ['day', 'hosp', 'icu', 'recovered', 'death']].reset_index(drop=True)
gender_df_f = gender_df.loc[gender_df['gender'] == 2, ['day', 'hosp', 'icu', 'recovered', 'death']].reset_index(drop=True)
gender_df = pd.merge(gender_df_h, gender_df_f, on='day', how='inner', suffixes=('_h', '_f'))
# Rename states
for state in states:
    gender_df[f'{state}_ph'] = gender_df[f'{state}_h'] / (gender_df[f'{state}_h'] + gender_df[f'{state}_f'])
# Select columns
cols = ['day'] + [f'{state}_ph' for state in states]
gender_df = gender_df.loc[:, cols]
# Fix 'day' columns
gender_df['day'] = gender_df['day'].str.replace('/', '-', regex=False)

### 3. Process age
# rename columns
age_df = pd.read_csv(os.path.join(path, 'age.csv'), sep=';')
age_df = age_df.rename(columns={'cl_age90': 'age', 'jour': 'day', 'rea': 'icu', 'rad': 'recovered', 'dc': 'death'})
# remove total statistics
age_df = age_df.loc[age_df['age'] != 0]
# map to our age classes
mapping_age = {9: 0, 19: 0, 29: 20, 39: 30, 49: 40, 59: 50, 69: 60, 79: 70, 89: 80, 90: 80}
age_df['age'] = age_df['age'].apply(lambda x: mapping_age.get(x))
# sum groupby
age_df['day'] = age_df['day'].str.replace('/', '-', regex=False)
age_df = age_df.groupby(['day', 'age'])[['hosp', 'icu', 'recovered', 'death']].agg('sum').reset_index()

### 4. Join gender and age
df = pd.merge(age_df, gender_df, on='day')

for state in states:
    df[f'{state}_h'] = df[f'{state}'] * df[f'{state}_ph']
    df[f'{state}_h'] = df[f'{state}_h'].apply(lambda x: round(x))
    df[f'{state}_f'] = df[f'{state}'] * (1 - df[f'{state}_ph'])
    df[f'{state}_f'] = df[f'{state}_f'].apply(lambda x: round(x))

df_ = None
for gender in ['h', 'f']:
    cols_gender = [f'{state}_{gender}' for state in states]
    df_gender = df.loc[:, ['age', 'day'] + cols_gender]
    df_gender['gender'] = gender
    df_gender = df_gender.rename(columns={f'{state}_{gender}': state for state in states})
    df_ = df_gender if df_ is None else df_.append(df_gender, ignore_index=True)

df = df_.sort_values(by=['day', 'age', 'gender']).reset_index(drop=True)
df['demography'] = df.apply(lambda x: f"{x['gender']}_{x['age']}", 1)
df['day'] = pd.to_datetime(df['day'], format='%Y-%m-%d')


### 5. Process new hospitalisations
new_cases_df = pd.read_csv(os.path.join(path, 'new.csv'), sep=';')
new_cases_df = new_cases_df.groupby('jour')['incid_hosp'].agg('sum').reset_index()
new_cases_df = new_cases_df.rename(columns={'jour': 'day', 'incid_hosp': 'new_hosp'})
# Cutting the first n days according to shift
shifted_new_cases_df = new_cases_df.loc[shift:, ['new_hosp']].reset_index(drop=True)
# Smoothing
x = shifted_new_cases_df['new_hosp'].values
smoothed = np.convolve(x, np.ones((smoothing,))/smoothing, mode='same')
shifted_new_cases_df['new_hosp'] = smoothed
n_demographies = df['demography'].nunique()
shifted_new_cases_df = shifted_new_cases_df.loc[shifted_new_cases_df.index.repeat(n_demographies)].reset_index(drop=True)
# cut the the last days according to `shift` in `df`
last_index = df.shape[0] - n_demographies * shift - 1
shifted_df = df.loc[:last_index,:]
# merge new and total df
shifted_df = pd.concat([shifted_df, shifted_new_cases_df], axis=1)

### 6. Compute the proportion for each demography of the daily current hospitalisation
total_day = shifted_df.groupby(['day'])['hosp'].agg('sum').reset_index()
total_day = total_day.rename(columns={'hosp': 'total_hosp'})
shifted_df = pd.merge(shifted_df, total_day, on='day', how='left')
shifted_df['prop_hosp'] = shifted_df['hosp'] / shifted_df['total_hosp']
# Assume same proportion of new hospitalisations 7 days later for each demography
shifted_df['part_new_hosp'] = shifted_df['new_hosp'] * shifted_df['prop_hosp']

# Load transition probability
transition_df = pd.read_csv(os.path.join(path, 'df_transitions_finale.csv'))
# "infected_hosp" is the probability of getting hospitalised given infected
transition_df = transition_df.loc[:, ['demography', 'infected_hosp']]
# Join transition probabilities with `shifted_df`
shifted_df = pd.merge(shifted_df, transition_df, on='demography', how='left')
shifted_df['infected'] = shifted_df['part_new_hosp'] / shifted_df['infected_hosp']
shifted_df['infected'] = shifted_df['infected'].apply(lambda x: round(x))
# Keep only relevant columns
cols_to_keep = ['day', 'demography', 'infected', 'hosp', 'icu', 'recovered', 'death']
df = shifted_df.loc[:, cols_to_keep]


### Asymptomatic
last_index = df.shape[0] - n_demographies * shift_asymp - 1
shifted_df = df.loc[:last_index,:]

shifted_infected = df.loc[shift_asymp * n_demographies:, ['infected']].reset_index(drop=True)
shifted_infected = shifted_infected.rename(columns={'infected': 'shifted_infected'})

shifted_df = pd.concat([shifted_df, shifted_infected], axis=1)
shifted_df['asymptomatic'] = shifted_df['shifted_infected'] * (1 / p_asympinf)

cols_to_keep = ['day', 'demography', 'asymptomatic', 'infected', 'hosp', 'icu', 'recovered', 'death']
df = shifted_df.loc[:, cols_to_keep]

df['day'] = pd.to_datetime(df['day'], format='%Y-%m-%d')


###
test_df = df.groupby(['day'])[['asymptomatic', 'infected', 'hosp', 'icu', 'recovered', 'death']].agg('sum').reset_index()
print(test_df.head(100))