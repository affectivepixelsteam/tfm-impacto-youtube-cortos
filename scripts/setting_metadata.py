#!/usr/bin/env python3


import pandas as pd
import json
import os
from pandas.io.json import json_normalize


def set_new_metadata(json_file):
    important_metadata_header = ['id', 'upload_date', 'categories', 'tags', 'age_limit', 'view_count', 'like_count',
                                 'dislike_count', 'average_rating', 'playlist_index']
    with open(json_file) as json_data:
        data = json.loads(json_data.read())
        importing_row = [data['id'], data['upload_date'], data['categories'], data['tags'], data['age_limit'],
                         data['view_count'], data['like_count'], data['dislike_count'], data['average_rating'],
                         data['playlist_index']]


    new_metadata = {
        important_metadata_header[0]: importing_row[0],
        important_metadata_header[1]: importing_row[1],
        important_metadata_header[2]: importing_row[2],
        important_metadata_header[3]: importing_row[3],
        important_metadata_header[4]: importing_row[4],
        important_metadata_header[5]: importing_row[5],
        important_metadata_header[6]: importing_row[6],
        important_metadata_header[7]: importing_row[7],
        important_metadata_header[8]: importing_row[8],
        important_metadata_header[9]: importing_row[9]
    }
    return new_metadata


years = ['2014', '2015', '2016', '2017', '2018']

for y in years:

    folder_path = os.path.join('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES', y)
    dirs = os.listdir(folder_path)
    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        json_file = directorio + '.info.json'
        new_json = 'new_' + json_file
        new_metadata = set_new_metadata(json_file)
        json.dump(new_metadata, new_json)