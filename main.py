import pandas as pd

from os.path import join
from os import listdir
from re import match, sub

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


def main():
    print(read_datasets())

if __name__ == "__main__":
    main()