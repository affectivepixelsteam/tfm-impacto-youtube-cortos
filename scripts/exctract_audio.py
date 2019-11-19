#!/usr/bin/env python3

import os


years = ['2014', '2015', '2016', '2017', '2018']

for y in years:

    folder_path = os.path.join('/mnt/RESOURCES/tfm-impacto-youtube-cortos/DATABASES', y)
    dirs = os.listdir(folder_path)

    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        video = directorio + '.mp4'
        cmd = 'ffmpeg -i ' + video + ' ' + directorio + '.wav'
        os.system(cmd)
