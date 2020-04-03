import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import pystan
from scipy.stats import gamma
from statsmodels.distributions.empirical_distribution import ECDF
from functools import reduce
from collections import OrderedDict
import dill as pickle

df = pd.read_csv('../data/COVID-19-up-to-date.csv',
                 encoding="ISO-8859-1",
                 parse_dates=['dateRep'],
                 dayfirst=True)

ifr = pd.read_csv('../data/weighted_fatality.csv', index_col=1).iloc[:, 1:]
ifr.index = ifr.index.str.replace(' ', '_')

serial_interval = pd.read_csv('../data/serial_interval.csv', index_col=0)

covariate_dates = pd.read_csv('../data/interventions.csv', index_col=0).iloc[:11, :7]
covariate_dates = covariate_dates.apply(lambda x: pd.to_datetime(x, dayfirst=True))
for variable in covariate_dates.columns:
    # making all covariates that happen after lockdown to have same date as lockdown
    covariate_dates.loc[covariate_dates[variable] > covariate_dates['lockdown'], variable] = covariate_dates['lockdown']

countries = ["Denmark",
             "Italy",
             "Germany",
             "Spain",
             "United_Kingdom",
             "France",
             "Norway",
             "Belgium",
             "Austria",
             "Sweden",
             "Switzerland"]

N2 = 75

countries_to_try = countries
# countries_to_try = ["Denmark", "Italy"]

N_days = []

data_per_country = OrderedDict()

for country in countries_to_try:

    print(f"Country: {country}")

    IFR = ifr.loc[country, 'weighted_fatality']

    covariates1 = covariate_dates.loc[country]

    d1 = df[df['countriesAndTerritories'] == country].sort_values(by='dateRep')

    idx = np.where(d1['cases'] > 0)[0][0]
    idx1 = np.where(d1['deaths'].cumsum() >= 10)[0][0]
    idx2 = idx1 - 30

    print(f"First non-zero cases is on day {idx + 1}, and 30 days before 10 deaths is day {idx2 + 1}")

    d1 = d1.iloc[idx2:, :]

    d1['days'] = (d1['dateRep'].copy() - d1['dateRep'].iloc[0]).apply(lambda x: x.days)

    for cov, date in covariates1.items():
        d1[cov] = (d1['dateRep'] >= date).astype(int)

    N = len(d1)

    d1 = d1.set_index('days', drop=True).reindex(np.arange(75))

    print(f"{N} days of data")

    N_days.append(N)

    mean1, cv1 = 5.1, 0.86
    x1 = gamma(cv1 ** -2, scale=mean1 / cv1 ** -2).rvs(size=int(5E6))
    mean2, cv2 = 18.8, 0.45
    x2 = gamma(cv2 ** -2, scale=mean2 / cv2 ** -2).rvs(size=int(5E6))
    f = ECDF(x1 + x2)

    def convolution(u):
        return IFR * f(u)

    h = np.zeros(N2)

    h[0] = convolution(1.5) - convolution(0)
    for i in range(1, len(h)):
        h[i] = (convolution(i + 1 + .5) - convolution(i + 1 - .5)) / (1 - convolution(i + 1 - .5))
    s = np.zeros(N2)
    s[0] = 1
    for i in range(1, len(s)):
        s[i] = s[i - 1] * (1 - h[i - 1])

    d1['f'] = s * h

    data_per_country[country] = d1

def extract_variable(var):
    return reduce(lambda x, y: x.join(y, how='outer'), [v[[var]].rename({var: k}, axis='columns')
                                                        for k, v in data_per_country.items()])


cases = extract_variable('cases').fillna(-1).astype(int)
deaths = extract_variable('deaths').fillna(-1).astype(int)
fs = extract_variable('f')

covs = dict()
for cov in covariate_dates.columns:
    covs[cov] = extract_variable(cov).reindex(np.arange(75)).ffill().astype(int)

covariate1 = covs['schools_universities']
covariate2 = covs['self_isolating_if_ill']
covariate3 = covs['public_events']
covariate4 = covs['schools_universities']|covs['public_events']|covs['lockdown']|covs['social_distancing_encouraged']|covs['self_isolating_if_ill']
covariate5 = covs['lockdown']
covariate6 = covs['social_distancing_encouraged']

stan_data = {'M': len(countries_to_try),  # Number of countries
             'N0': 6,  # Number of days to impute imperfections
             'N': N_days,
             'N2': N2,
             'x': np.linspace(1, N2, num=N2),
             'cases': cases.values,
             'deaths': deaths.values,
             'f': fs.values,
             'covariate1': covariate1.values,
             'covariate2': covariate2.values,
             'covariate3': covariate3.values,
             'covariate4': covariate4.values,
             'covariate5': covariate5.values,
             'covariate6': covariate6.values,
             'EpidemicStart': [31 for _ in range(len(countries_to_try))],
             'SI': serial_interval['fit'].iloc[:N2].values}

with open('python_stan_data.pkl', 'wb') as file:
    pickle.dump(stan_data, file)

# sm = pystan.StanModel(file='../stan-models/base.stan')
# print('Done compiling STAN model...')
#
# fit = sm.sampling(data=stan_data, verbose=True,
#                   iter=200, warmup=100, chains=4, thin=4,
#                   control={'adapt_delta': 0.90, 'max_treedepth': 10})
