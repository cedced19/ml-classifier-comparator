import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

# select specific datasets
import sys
default_path = ''
if (len(sys.argv) != 1):
    default_path = sys.argv[1]


dataset = pd.read_csv('datasets/' + default_path + '.csv')

array = np.load('results/' + default_path + '.npy', allow_pickle=True)

recall_list = []
precision_list = []
test_score_list = []

for experiment in array:
    recall_list.append(experiment['recall_score'])
    precision_list.append(experiment['precision_score'])
    test_score_list.append(experiment['test_score'])

plt.figure()
plt.xlabel(r'Score')
plt.ylabel(r'Recall - Rappel $\frac{TP}{TP+FN}$')
plt.scatter(test_score_list, recall_list)
plt.show()

plt.figure()
plt.xlabel(r'Recall - Rappel $\frac{TP}{TP+FN}$')
plt.ylabel(r'Precision $\frac{TP}{TP+FP}$')
plt.scatter(recall_list, precision_list)
plt.show()

