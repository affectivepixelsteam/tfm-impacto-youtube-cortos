import pandas as pd
import csv
import os
import json

years = ['2014', '2015', '2016', '2017', '2018']

for y in years:

    folder_path = os.path.join('/mnt/RESOURCES/tfm-impacto-youtube-cortos/DATABASES', y)
    dirs = os.listdir(folder_path)
    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        file = directorio + '.csv'
        json_file = directorio + '.info.json'
        df = pd.read_csv(file)
        with open(json_file) as json_data:
            data = json.loads(json_data.read())
            importing_row = [data['description']]
        description = importing_row
        df['Description'] = description
        df.to_csv(file)