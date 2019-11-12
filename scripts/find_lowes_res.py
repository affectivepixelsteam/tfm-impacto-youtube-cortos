#!/usr/bin/env python3

import cv2
import os
import pandas as pd


alturas = []
anchuras = []
fotograms = []
ids = []
years = ['2014', '2015', '2016', '2017', '2018']

for y in years:

    folder_path = os.path.join('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES', y)
    dirs = os.listdir(folder_path)
    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)

        video_id = directorio

        videotoget = video_id + '.mp4'
        video = cv2.VideoCapture(videotoget)
        width = video.get(3)
        height = video.get(4)
        fps = video.get(5)
        ids.append(directorio)
        alturas.append(int(height))
        anchuras.append(int(width))
        fotograms.append(fps)

data = {'Id' : ids, 'H' : alturas, 'W' : anchuras, 'FPS' : fotograms }
df = pd.DataFrame(data)

count_Heights = df['H'].value_counts()
print(count_Heights)

count_fps = df['FPS'].value_counts()
print(count_fps)


