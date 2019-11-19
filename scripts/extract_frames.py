#!/usr/bin/env python3

import shutil
import os


years = ['2014', '2015', '2016', '2017', '2018']

for y in years:

    folder_path = os.path.join('/mnt/RESOURCES/tfm-impacto-youtube-cortos/DATABASES', y)
    dirs = os.listdir(folder_path)

    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        video = directorio + '.mp4'
        cmd = 'ffmpeg -i ' + video + ' ' + directorio + '%d.jpg'
        os.system(cmd)
        name_new_dir = directorio + '_frames'
        path_to_folder = os.path.join(path_inside_directorio, name_new_dir)
        indirs = os.listdir(path_inside_directorio)
        if not os.path.exists(path_to_folder):
            os.mkdir(name_new_dir)
            for file in indirs:
                if file.endswith('.jpg'):
                    shutil.copy(file, name_new_dir)
                    os.remove(file)


