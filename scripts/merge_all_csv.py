import sys
import os
import numpy as np
import pandas as pd


years = ['2014', '2015', '2016']

for y in years:
    folder_path = os.path.join('/mnt/pgth04b/DATABASES_CRIS/FINALISTAS_ORIGINAL/DATABASES', y)
    dirs = os.listdir(folder_path)
    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        indirs = os.listdir(path_inside_directorio)
        for file in indirs:
            if file.endswith('total.csv'):
                print(file)
                data_merged = pd.read_csv('/mnt/pgth04b/DATABASES_CRIS/data_merged.csv')
                file_open = pd.read_csv(file)
                data_merged = pd.concat([file_open, data_merged])
                data_merged.to_csv('/mnt/pgth04b/DATABASES_CRIS/data_merged.csv')
