#!/usr/bin/env python3

import cv2
import os
import pandas as pd


alturas = []
fotograms = []
ids = []

years = ['2014', '2015', '2016', '2017', '2018']

for y in years:

    folder_path = os.path.join('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES', y)
    dirs = os.listdir(folder_path)

    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        indirs = os.listdir(path_inside_directorio)


        videonormal = directorio + '.mp4'

        video = cv2.VideoCapture(videonormal)
        height = int(video.get(4))
        if height > 480:
            print('HEEEEY')
        fps = video.get(5)
        ids.append(directorio)
        alturas.append(int(height))
        fotograms.append(fps)

data = {'Id': ids, 'H': alturas, 'FPS': fotograms}
df = pd.DataFrame(data)

print(df)

count_Heights = df['H'].value_counts()
print(count_Heights)

count_fps = df['FPS'].value_counts()
print(count_fps)


