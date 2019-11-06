#!/usr/bin/env python3

## Descarga comentarios de los videos y los almacena en la misma carpeta que el resto del material sobre el video que
## ya se tenía.
## Autor: Aitor Galán García
## 6 de Noviembre de 2019

import os

os.chdir("/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES")

years = ['2014', '2015', '2016', '2017', '2018']

for y in years:

    folder_path = os.path.join('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES', y)
    os.chdir(folder_path)
    dirs = os.listdir(folder_path)

    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        video_id = directorio
        name_comment = directorio + '_comment'
        cmd = '../../../descarga-comentarios/./downloader.py  --youtubeid ' + directorio + ' --output ' + name_comment
        os.system(cmd)