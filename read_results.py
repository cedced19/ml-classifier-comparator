import numpy as np
import pandas as pd

# select specific datasets
import sys
default_path = ''
if (len(sys.argv) != 1):
    default_path = sys.argv[1]

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

dataset = pd.read_csv('datasets/' + default_path + '.csv')

array = np.load('results/' + default_path + '.npy', allow_pickle=True)


for experiment in array:
    print(experiment)
