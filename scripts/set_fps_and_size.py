#!/usr/bin/env python3

import cv2
import os
import numpy as np
import pandas as pd

years = ['2014', '2015', '2016', '2017', '2018']

for y in years:

    folder_path = os.path.join('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES', y)
    dirs = os.listdir(folder_path)
    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        video_id = directorio
        video_in = video_id + '.mp4'
        video_fps = 'NEW_FPS_TWICE_' + video_id + '.mp4'
        os.chdir(path_inside_directorio)
        final = 'RESIZEDTWICE_' + video_id + '.mp4'
        if video_fps in path_inside_directorio:
            video = cv2.VideoCapture(video_fps)
            height = int(video.get(4))
            if height > 480:
                cmd = 'ffmpeg -i ' + video_fps + ' -vf scale=848:480 ' + final
                os.system(cmd)
        else:
            video = cv2.VideoCapture(video_in)
            height = int(video.get(4))
            if height > 480:
                cmd = 'ffmpeg -i ' + video_in + ' -vf scale=848:480 ' + final
                os.system(cmd)
