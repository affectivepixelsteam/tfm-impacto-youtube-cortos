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
        video_input = video_id + '.mp4'
        video_output = 'NEW_FPS_' + video_id + '.mp4'
        os.chdir(path_inside_directorio)
        cmd = 'ffmpeg -i ' + video_input + ' -filter:v fps=fps=24 ' + video_output
        os.system(cmd)
        video = cv2.VideoCapture(video_input)
        height = int(video.get(4))
        final = 'RESIZED_' + video_id + '.mp4'
        if height > 480:
            cmd2 = 'ffmpeg -i ' + video_output + ' -vf scale=848:480 ' + final
            os.system(cmd2)
