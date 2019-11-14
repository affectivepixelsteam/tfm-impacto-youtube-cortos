#!/usr/bin/env python3

import cv2
import os
import pandas as pd




years = ['2014', '2015', '2016', '2017', '2018']

for y in years:

    folder_path = os.path.join('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES', y)
    dirs = os.listdir(folder_path)

    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        indirs = os.listdir(path_inside_directorio)
        videoresized = 'RESIZEDTWICE_' + directorio + '.mp4'
        videofps = 'NEW_FPS_TWICE_' + directorio + '.mp4'
        if videoresized in indirs:

            src = videoresized
            dst = directorio + '.mp4'
            if dst in indirs:
                os.remove(dst)
            os.rename(src, dst)


