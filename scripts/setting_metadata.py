#!/usr/bin/env python3


import pandas as pd
import json
import os
from pandas.io.json import json_normalize
import csv


def set_new_metadata_json(json_file):
    important_metadata_header = ['id', 'upload_date', 'categories', 'tags', 'age_limit', 'view_count', 'like_count',
                                 'dislike_count', 'average_rating', 'playlist_index', 'duration']
    with open(json_file) as json_data:
        data = json.loads(json_data.read())
        importing_row = [data['id'], data['upload_date'], data['categories'], data['tags'], data['age_limit'],
                         data['view_count'], data['like_count'], data['dislike_count'], data['average_rating'],
                         data['playlist_index'], data['duration']]


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
        important_metadata_header[9]: importing_row[9],
        important_metadata_header[10]: importing_row[10]
    }
    return new_metadata

def set_new_metadata_csv(csv_file):
    important_metadata_header = ['id', 'upload_date', 'categories', 'tags', 'age_limit', 'view_count', 'like_count',
                                 'dislike_count', 'average_rating', 'playlist_index', 'duration']
    with open(json_file) as json_data:
        data = json.loads(json_data.read())
        importing_row = [data['id'], data['upload_date'], data['categories'], data['tags'], data['age_limit'],
                         data['view_count'], data['like_count'], data['dislike_count'], data['average_rating'],
                         data['playlist_index'], data['duration']]
    new_metadata = [[important_metadata_header[0], important_metadata_header[1], important_metadata_header[2],
                     important_metadata_header[3], important_metadata_header[4], important_metadata_header[5],
                     important_metadata_header[6], important_metadata_header[7], important_metadata_header[8],
                     important_metadata_header[9], important_metadata_header[10]],
                    [importing_row[0], importing_row[1], importing_row[2], importing_row[3], importing_row[4],
                     importing_row[5], importing_row[6], importing_row[7], importing_row[8], importing_row[9],
                     importing_row[10]]]
    return new_metadata

type = input("CSV or JSON?")
years = ['2014', '2015', '2016', '2017', '2018']

for y in years:

    folder_path = os.path.join('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES', y)
    dirs = os.listdir(folder_path)
    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        json_file = directorio + '.info.json'


        if type == 'CSV' or type == 'csv':
            new_csv =  directorio + '.csv'
            new_metadata = set_new_metadata_csv(json_file)

            with open(new_csv, 'w+') as csv_file_end:
                writer = csv.writer(csv_file_end)
                writer.writerows(new_metadata)
            csv_file_end.close()

        elif type == 'JSON' or type == 'json':
            new_json = 'new_' + json_file
            new_metadata = set_new_metadata_json(json_file)
            with open(new_json, 'w+') as json_file_end:
                json.dump(new_metadata, json_file_end)

        else:
            print('wrong type')