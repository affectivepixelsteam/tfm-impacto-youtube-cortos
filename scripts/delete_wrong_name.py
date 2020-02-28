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
            if file.endswith('complete.csv_total.csv'):
                print('a')
                os.remove(file)