import os
import shutil

years = ['2014', '2015', '2016', '2017', '2018']

os.mkdir("/home/aitorgalan/Escritorio/all_metadata")
os.mkdir("/home/aitorgalan/Escritorio/all_metadata/json_metadata")
os.mkdir("/home/aitorgalan/Escritorio/all_metadata/csv")
os.mkdir("/home/aitorgalan/Escritorio/all_metadata/json_metadata/all_json_metadata")
os.mkdir("/home/aitorgalan/Escritorio/all_metadata/json_metadata/only_relevant_json_metadata")

path_to_csv = "/home/aitorgalan/Escritorio/all_metadata/csv"
path_to_all_json = "/home/aitorgalan/Escritorio/all_metadata/json_metadata/all_json_metadata"
path_to_relevant_json = "/home/aitorgalan/Escritorio/all_metadata/json_metadata/only_relevant_json_metadata"

for y in years:

    folder_path = os.path.join('/mnt/RESOURCES/tfm-impacto-youtube-cortos/DATABASES', y)
    dirs = os.listdir(folder_path)

    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        indirs = os.listdir(path_inside_directorio)
        all_json_name = directorio + '.info.json'
        relevant_json_name = 'new_' + all_json_name
        csv_name = directorio + '.csv'
        for file in indirs:
            if file == csv_name:
                shutil.copy(file, path_to_csv)
            elif file == all_json_name:
                shutil.copy(file, path_to_all_json)
            elif file == relevant_json_name:
                shutil.copy(file, path_to_relevant_json)
            else:
                continue
