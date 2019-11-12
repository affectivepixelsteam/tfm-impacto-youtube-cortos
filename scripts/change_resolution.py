#!/usr/bin/env python3

import cv2
import os

os.chdir('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES')

years = ['2014', '2015', '2016', '2017', '2018']

def resolution_video(videoid):
    videotoget = videoid + '.mp4'
    video = cv2.VideoCapture(videotoget)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    return [int(height), int(width)]


def change_resolution(videoid, new_res):
    videotochange = videoid + '.mp4'


for y in years:

    folder_path = os.path.join('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES', y)
    os.chdir(folder_path)
    dirs = os.listdir(folder_path)
    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        video_id = directorio
        resolution_of_video = resolution_video(video_id)
        if resolution_of_video[0] > 480:
            change_resolution(video_id, 480)
        elif resolution_of_video[0] < 480




