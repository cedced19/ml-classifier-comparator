import pandas as pd
import numpy as np
from progressbar import ProgressBar

from os.path import join
from os import listdir, urandom
from re import match, sub

from itertools import product
from sklearn.base import clone

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, confusion_matrix, roc_auc_score, recall_score

# Import methods for classifications and imbalanced dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, SMOTEN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

input_path = "datasets/"
output_path = "results/"

csv_separator = ','

def read_datasets():
    datasets = []
    csv_files = [csv_file for csv_file in listdir(input_path) if match('^.+\.csv$', csv_file)]
    for csv_file in csv_files:
        dataset = pd.read_csv(join(input_path, csv_file))
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

def check_classifiers(models):
    """Creates individual classifiers from parameters grid."""
    try:
        flat_models = []
        for model_name, model, *param_grid, param_fit in models:
            if param_grid == []:
                flat_models += [(model_name, model, param_fit)]
            else:
                
                flat_parameters = _flatten_parameters_list(param_grid[0])
                print(flat_parameters)
                for ind, parameters in enumerate(flat_parameters):
                    flat_models += [(model_name + str(ind + 1), model, param_fit)]
                    #flat_models += [(model_name + str(ind + 1), clone(model).set_params(**parameters), param_fit)]
    except:
        raise ValueError("The classifier should be a list of (classifier name, classifier) pairs or (classifier name, classifier, parameters grid) triplets.")
    return flat_models

def check_oversamplers(models):
    """Creates individual oversamplers from parameters grid."""
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
        raise ValueError("The oversampler should be a list of (oversampler name, oversampler) pairs or (oversampler name, oversampler, parameters grid) triplets.")
    return flat_models

def score_method(X_train, X_test, y_train, y_test, oversampler, classifier):
    y_predict = (classifier[1].predict(X_test.values)> 0.5).astype("int32").flatten()
    return {
        'train_score': classifier[1].score(X_train.values, y_train.values),
        'test_score': classifier[1].score(X_test.values, y_test.values),
        'classifier': classifier[1],
        'oversampler': oversampler[1],
        'cm': confusion_matrix(y_test.values, y_predict),
        'recall_score': recall_score(y_test.values, y_predict),
        'roc_auc_score': roc_auc_score(y_test.values, y_predict),
        'precision_score': precision_score(y_test.values, y_predict),
        'f1_score': f1_score(y_test.values, y_predict),
        'f1_score_macro': f1_score(y_test.values, y_predict, average='macro'),
        'f1_score_weighted': f1_score(y_test.values, y_predict, average='weighted')
    }

def average_score_method(results):
    blacklist = ['classifier', 'oversampler']
    l = len(results)
    d = {}
    for i in results[0].keys():
        if (i in blacklist):
            d[i] = results[0][i]
        else:
            tmp = 0
            for j in range(l):
                tmp += results[j][i]
            tmp = tmp/l
            d[i] = tmp
    return d

def execute_methods(datasets, random_states, classifiers, oversampling_methods, scoring):
    print("Executing methods")
    classifiers = check_classifiers(classifiers)
    oversampling_methods = check_oversamplers(oversampling_methods)
    n_random_states = len(random_states)
    max_iter = n_random_states * len(datasets) * len(oversampling_methods) * len(classifiers)
    progress_bar = ProgressBar(redirect_stdout=False, max_value=max_iter)
    iterations = 0
    all_results = []
    for dataset_name, (X, y) in datasets:
        results = []
        for classifier in classifiers:
            # get args for model.fit
            argv_fit = {}
            if (len(classifier) >= 3):
                argv_fit = classifier[2]

            for oversampling_method in oversampling_methods:
                #print("\nDataset: ", dataset_name, ", oversampling method:", oversampling_method[0], ", classifier: ", classifier[0], "\n")
                tmp_results = []
                for random_state in random_states:
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                        
                        if oversampling_method[1] is not None:
                            if ('random_state' in oversampling_method[1].get_params().keys()):
                                oversampling_method[1].set_params(random_state=random_state)
                            X_train, y_train = oversampling_method[1].fit_resample(X_train.to_numpy(), y_train.to_numpy())
                            X_train = pd.DataFrame(X_train)
                            y_train = pd.DataFrame(y_train)
                        

                        if ('random_state' in classifier[1].get_params().keys()):
                                classifier[1].set_params(random_state=random_state)
                        
                        # if (len(classifier) == 5):
                            # classifier[1].compile(*(classifier[4]))

                        classifier[1].fit(X_train.values, y_train.values.ravel(), **argv_fit)
                        tmp_results.append(score_method(X_train, X_test, y_train, y_test, oversampling_method, classifier))
                        #print('train score:', classifier[1].score(X_train.values, y_train.values))
                        #print('test score:', classifier[1].score(X_test.values, y_test.values))
                    except Exception as e:
                        print('Error:', e)
                        print('Oversampler:', oversampling_method)
                        print('Classifier:', classifier)
                        exit() 
                    iterations += 1
                    progress_bar.update(iterations)
                results.append(average_score_method(tmp_results))
        np.save('{}/{}.npy'.format(output_path, dataset_name), results)
        all_results.append((dataset_name, results))
        print(all_results)                 


def main():
    print("Defining parameters")
    # Read dataset
    datasets = read_datasets()
    # Get random states
    n_random_states = 10
    random_states = select_random_states(n_random_states)

    debug = True

    if debug:
        
        model = Sequential()
        model.add(Dense(25, activation='relu')) 
        model.add(Dense(20, activation='relu')) 
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid')) # la sortie est forcément un softmax du nombre de classes
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        classifiers = [
            (
                'Custom', model,
                [{}],
                {'epochs':100, 'verbose':1}
            )
        ]


        oversampling_methods = [
            ('None',None),
            ('RandomOverSampler', RandomOverSampler()),
        ] 

        """ classifiers = [
            (
                'RandomForestClassifier', RandomForestClassifier(),
                [{
                    'n_estimators': [50, 100, 200],
                    'max_depth': [4, 10, None],
                    'criterion' : ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2']
                }]
            )
        ]

        oversampling_methods = [
            ('None',None),
            ('RandomOverSampler', RandomOverSampler()),
        ] """
    else:
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
            ),(
                'RandomForestClassifier', RandomForestClassifier(),
                [{
                    'n_estimators': [50, 100, 200],
                    'max_depth': [4, 10, None],
                    'criterion' : ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2']
                }]
            )
        ]

        oversampling_methods = [
            ('None',None),
            ('RandomOverSampler', RandomOverSampler()),
            ('RandomUnderSampler', RandomUnderSampler()),
            (
                'SMOTE', SMOTE(),
                [{
                    'k_neighbors': [3,5,20]
                }]
            ),
            (
                'SMOTEN', SMOTEN(),
                [{
                    'k_neighbors': [3,5,20]
                }]
            ),
            (
                'B1-SMOTE', BorderlineSMOTE(kind='borderline-1'),
                [{
                    'k_neighbors': [3,5,20]
                }]
            ),
            (
                'B2-SMOTE', BorderlineSMOTE(kind='borderline-2'),
                [{
                    'k_neighbors': [3,5,20]
                }]
            )
        ]

    scoring = [] # to be coded later

    execute_methods(datasets, random_states, classifiers, oversampling_methods, scoring)


if __name__ == "__main__":
    main()