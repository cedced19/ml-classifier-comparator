import numpy as np
import pandas as pd

dataset = pd.read_csv('datasets/brazil_dataset_french_clean.csv')

unique, counts = np.unique(dataset.iloc[:, -1], return_counts=True)
amount = dict(zip(unique, counts))
total = sum(counts)
print('Total: ' + str(total))
for key, value in amount.items():
    print(str(key) + ' : ' + str(value) + ' , ' + str(round(value/total, 4)))

array = np.load('results/brazil_dataset_french_clean.npy', allow_pickle=True)

max_score = 0
max_score_method = None
for experiment in array:
    if (experiment['test_score'] >= max_score):
        max_score = experiment['test_score']
        max_score_method = experiment

print("\nMax score\n")
print('Score: ' + str(max_score_method['test_score']) + ' Recall: ' + str(max_score_method['recall_score']))
print("\n")
print(max_score_method['cm'])
print("\n")
print(max_score_method)

max_recall = 0
max_recall_method = None
for experiment in array:
    if (experiment['recall_score'] >= max_recall):
        max_recall = experiment['recall_score']
        max_recall_method = experiment

print("\nMax recall\n")
print('Score: ' + str(max_recall_method['test_score']) + ' Recall: ' + str(max_recall_method['recall_score']))
print("\n")
print(max_recall_method['cm'])
print("\n")
print(max_recall_method)
