import os

import zipfile
import pandas as pd
import shutil, json, csv
import cv2

def create_folder(path_of_new_folder):
    if not os.path.exists(path_of_new_folder):
        os.makedirs(path_of_new_folder)

def list_files_in_dir(path_of_directory):
    return os.listdir(path_of_directory)


def extract_all_from_zip(path_to_zip_file, dir_to_extract):
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(dir_to_extract)
    zip_ref.close()


def extract_files_from_subfolders(input_root_folder, name_folder_2_extract,folder_extension_2_extract, output_path):
    folder_options = ["FINALISTAS_ORIGINAL", "NO_FINALISTAS"]
    for video_position in folder_options:
        root_path_positions = os.path.join(input_root_folder, video_position)
        if("NO_FINALISTAS" in video_position):
            years = ["2017", "2018", "2019"]
        else:
            years = ["2014", "2015", "2016", "2017", "2018", "2019"]
        for year in years:
            path_with_year = os.path.join(root_path_positions, "DATABASES", year)
            list_video_ids = os.listdir(path_with_year)
            for video_id in list_video_ids:
                path_files_in_folder = os.path.join(path_with_year, video_id, name_folder_2_extract)
                files_in_folder = os.listdir(path_files_in_folder)
                for file in files_in_folder:
                    if(folder_extension_2_extract in file):
                        path_file = os.path.join(path_files_in_folder, file)
                        out_path = os.path.join(output_path, video_position,"DATABASES",year,video_id,name_folder_2_extract)
                        if(not os.path.exists(out_path)):
                            os.makedirs(out_path)
                        shutil.copy(path_file, out_path)

def insert_folders_in_DS(input_root_folder, name_folder_2_insert,folder_extension_2_insert, root_path_2_insert_data):
    folder_options = ["FINALISTAS_ORIGINAL", "NO_FINALISTAS"]
    for video_position in folder_options:
        root_path_positions = os.path.join(input_root_folder, video_position)
        if ("NO_FINALISTAS" in video_position):
            years = ["2017", "2018", "2019"]
        else:
            years = ["2014", "2015", "2016", "2017", "2018", "2019"]
        for year in years:
            path_with_year = os.path.join(root_path_positions, "DATABASES", year)
            list_video_ids = os.listdir(path_with_year)
            for video_id in list_video_ids:
                path_files_in_folder = os.path.join(path_with_year, video_id, name_folder_2_insert)
                files_in_folder = os.listdir(path_files_in_folder)
                for file in files_in_folder:
                    if (folder_extension_2_insert in file):
                        path_file = os.path.join(path_files_in_folder, file)
                        out_path = os.path.join(root_path_2_insert_data, video_position, "DATABASES", year, video_id,name_folder_2_insert)
                        if (not os.path.exists(out_path)):
                            #print("the destination folder: ", out_path, " doesn't exist")
                            os.makedirs(out_path)
                        shutil.copy(path_file, out_path)



root_path = "/mnt/pgth04b/root_path_2_insert_data"
name_folder_2_extract = ""
folder_extension_2_extract = "_24fps_854wx480h.mp4"
output_path = "/mnt/pgth04b/reduced_videos"
#extract_files_from_subfolders(root_path, name_folder_2_extract,folder_extension_2_extract, output_path)
insert_folders_in_DS(input_root_folder=output_path, name_folder_2_insert=name_folder_2_extract,folder_extension_2_insert=folder_extension_2_extract, root_path_2_insert_data=root_path)
