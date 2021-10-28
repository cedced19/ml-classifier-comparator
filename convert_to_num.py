import pandas as pd
from sklearn import preprocessing

from os.path import join
from os import listdir
from re import match

input = "datasets_string/"
output = "datasets/"

valid_cols = ['Genre']
to_scale_cols = ['Moyenne']
to_convert_cols = ['Departement', 'Origine', 'Mobilite']
output_col = ['Bourse']

csv_files = [csv_file for csv_file in listdir(input) if match('^.+\.csv$', csv_file)]

for csv_file in csv_files:
    df=pd.read_csv(join(input, csv_file), sep=',')
    df = df.apply(preprocessing.LabelEncoder().fit_transform)

    A = df[valid_cols]

    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler
    encoder = OneHotEncoder(sparse=False)
    B = encoder.fit_transform(df[to_convert_cols])

    C = MinMaxScaler().fit_transform(df[to_scale_cols])
    #C = StandardScaler().fit_transform(df[to_scale_cols])
    #C = RobustScaler().fit_transform(df[to_scale_cols])

    B = pd.DataFrame(B)
    C = pd.DataFrame(C)
    Y = df[[output_col]]
    newdf = pd.concat([A,B,C,Y], axis=1)
    newdf.to_csv(join(output, csv_file), sep=',', encoding='utf-8')
    print(newdf)