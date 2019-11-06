#!/usr/bin/env python3

## Organiza todos los videos descargados y sus metadatos en carpetas con el nombre del id del video.
## Autor: Aitor Galán García
## 5 de Noviembre de 2019

import os
import shutil

os.chdir("/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES")

years = ['2014', '2015', '2016', '2017', '2018']


for y in years:

    folder_path = os.path.join('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES', y)
    os.chdir(folder_path)
    dirs = os.listdir(folder_path)

    for file in dirs:

        folderName = file[0:11]
        path_to_folder = os.path.join(folder_path, folderName)
        file_to_copy = os.path.join(folder_path, file)

        if not os.path.exists(path_to_folder):
            os.mkdir(folderName)
            shutil.copy(file_to_copy, folderName)

        else:
            shutil.copy(file_to_copy, folderName)
