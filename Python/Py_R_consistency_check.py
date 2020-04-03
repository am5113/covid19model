import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dill as pickle

import rpy2.robjects as robjects

readRDS = robjects.r['readRDS']
r_stan_data = readRDS('r_stan_data.rds')

with open('python_stan_data.pkl', 'rb') as file:
    py_stan_data = pickle.load(file)

py_stan_data.keys()


def mycheck(key):
    r_obj = np.array(r_stan_data.rx2(key))
    py_obj = np.array(py_stan_data[key])

    # Special cases
    if key == 'covariate4':  # R saves this variable transposed
        r_obj = r_obj.T

    if key == 'f' or key == 'SI':
        checker = lambda x, y: np.allclose(x, y, atol=1e-5, rtol=0)
    else:
        checker = lambda x, y: np.all(x == y)

    if checker(r_obj, py_obj):
        print(f"Object {key} is fine...")
    else:
        print(f"Found a problem with {key}!")


for key in py_stan_data.keys():
    mycheck(key)

fig, ax = plt.subplots(1, 2, dpi=150, figsize=(10,2))
ax[0].plot(np.array(r_stan_data.rx2('f')))
ax[1].plot(np.array(py_stan_data['f']))
plt.tight_layout()
plt.show()
