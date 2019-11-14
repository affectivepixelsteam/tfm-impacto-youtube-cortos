#!/usr/bin/env python3

import cv2
import os
import numpy as np
import pandas as pd

years = ['2014', '2015', '2016', '2017', '2018']

big = []

for y in years:

    folder_path = os.path.join('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES', y)
    dirs = os.listdir(folder_path)
    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        video_id = directorio
        video_in = video_id + '.mp4'
        os.chdir(path_inside_directorio)
        video = cv2.VideoCapture(video_in)
        height = int(video.get(4))
        video_out = 'out_' + video_in
        if height > 480:
            cmd = 'ffmpeg -i ' + video_in + ' -vf scale=848:480 ' + video_out
            os.system(cmd)
            src = video_out
            dst = video_in
            os.remove(dst)
            os.rename(src, dst)


print(big)
length_of_big = len(big)
print(length_of_big)

