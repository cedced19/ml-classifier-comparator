import numpy as np

array = np.load('results/brazil_dataset_french_clean.npy', allow_pickle=True)

max_score = 0
max_score_method = None
for experiment in array:
    if (experiment['test_score'] >= max_score):
        max_score = experiment['test_score']
        max_score_method = experiment

print(max_score_method)
