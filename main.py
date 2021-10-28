import pandas as pd
import numpy as np
from progressbar import ProgressBar

from os.path import join
from os import listdir, urandom
from re import match, sub

from itertools import product
from sklearn.utils import check_X_y, check_random_state
from sklearn.base import clone

from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold


# Import methods for classifications and imbalanced dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler


output_path = "datasets/"

csv_separator = ','

def read_datasets():
    datasets = []
    csv_files = [csv_file for csv_file in listdir(output_path) if match('^.+\.csv$', csv_file)]
    for csv_file in csv_files:
        dataset = pd.read_csv(join(output_path, csv_file))
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        dataset_name = sub(".csv", "", csv_file)
        datasets.append((dataset_name, (X, y)))
    return datasets

def select_random_states(n):
    arr = []
    for i in range(n):
        arr.append(int(urandom(1)[0] / 255 * (2**32)))
    return arr

def _flatten_parameters_list(parameters_list):
    """Flattens a dictionaries' list of parameters."""
    flat_parameters = []
    for parameters_dict in parameters_list:
        flat_parameters += [dict(zip(parameters_dict.keys(), parameter_product)) for parameter_product in product(*parameters_dict.values())]
    return flat_parameters

def check_models(models, model_type):
    """Creates individual classifiers and oversamplers from parameters grid."""
    try:
        flat_models = []
        for model_name, model, *param_grid in models:
            if param_grid == []:
                flat_models += [(model_name, model)]
            else:
                flat_parameters = _flatten_parameters_list(param_grid[0])
                for ind, parameters in enumerate(flat_parameters):
                    flat_models += [(model_name + str(ind + 1), clone(model).set_params(**parameters))]
    except:
        raise ValueError("The {model_type}s should be a list of ({model_type} name, {model_type}) pairs or ({model_type} name, {model_type}, parameters grid) triplets.".format(model_type=model_type))
    return flat_models

def execute_methods(datasets, random_states, classifiers, oversampling_methods, scoring):
    print("Executing methods")
    classifiers = check_models(classifiers, "classifier")
    oversampling_methods = check_models(oversampling_methods, "oversampler")
    max_iter = len(random_states) * len(datasets) * len(oversampling_methods) * len(classifiers)
    progress_bar = ProgressBar(redirect_stdout=False, max_value=max_iter)
    iterations = 0
    for dataset_name, (X, y) in datasets:
        for classifier in classifiers:
            for oversampling_method in oversampling_methods:
                print("Dataset: ", dataset_name, ", oversampling method:", oversampling_method[0], ", classifier: ", classifier[0])
                for random_state in random_states:
                    method = None
                    classifier[1].set_params(random_state=random_state)
                    if oversampling_method[1] is not None:
                        oversampling_method[1].set_params(random_state=random_state)
                        method = Pipeline([oversampling_method, classifier])
                    else:
                        method = Pipeline([classifier])
                    cv = StratifiedKFold(n_splits=2, random_state=random_state, shuffle=True)
                    scores = cross_validate(method, X, y, cv=cv, scoring=scoring)
                    print(scores)
                    iterations += 1
                    progress_bar.update(iterations)                    


def main(n_random_states = 5):
    print("Defining parameters")
    # Read dataset
    datasets = read_datasets()
    # Get random states
    random_states = select_random_states(n_random_states)

    classifiers = [
        (
            'GBM',GradientBoostingClassifier(),
            [{
                'n_estimators': [50, 100, 200]
            }]
        ),(
            'KNN',KNeighborsClassifier(),
            [{
                'n_neighbors': [3,5,8]
            }]
        )
    ]

    oversampling_methods = [
        ('None',None),
        ('RandomOverSampler', RandomOverSampler()),
        (
            'SMOTE', SMOTE(),
            [{
                'k_neighbors': [3,5,20]
            }]
        ),
        (
            'B1-SMOTE', BorderlineSMOTE(kind='borderline1'),
            [{
                'k_neighbors': [3,5,20]
            }]
        ),
        (
            'B2-SMOTE', BorderlineSMOTE(kind='borderline2'),
            [{
                'k_neighbors': [3,5,20]
            }]
        ),
        (
            'KMeansSMOTE', KMeansSMOTE(),
            [
                {
                    'cluster_balance_threshold': [1,float('Inf')],
                    'density_exponent': [0, 2, None], 
                    'k_neighbors': [3,5,20, float('Inf')],
                    'kmeans_estimator': [2,20,50,100,250,500],
                    'n_jobs':[-1]
                },
                {
                    'cluster_balance_threshold': [float('Inf')],
                    'kmeans_estimator': [1],
                    'k_neighbors': [3,5],
                    'n_jobs':[-1]
                }
            ]
        )
    ]

    scoring = ['accuracy']

    execute_methods(datasets, random_states, classifiers, oversampling_methods, scoring)


if __name__ == "__main__":
    main()