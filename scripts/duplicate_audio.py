import shutil
import os


#years_fin = ['2014', '2015', '2016', '2017', '2018', '2019']

#for yf in years_fin:

#    folder_path = os.path.join('/mnt/pgth04b/DATABASES_CRIS/FINALISTAS_ORIGINAL/DATABASES', yf)
#    destino = os.path.join('/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas', yf)
#    dirs = os.listdir(folder_path)
#    for directorio in dirs:
#        path_inside_directorio = os.path.join(folder_path, directorio)
#        os.chdir(path_inside_directorio)
#        files = os.listdir(path_inside_directorio)
#        for f in files:
#            if f.endswith('.wav'):
#                dst = os.path.join(destino, f)
#                shutil.copy(f, dst)

#years_no_fin = ['2017', '2018', '2019']

#for ynf in years_no_fin:
#    folder_path = os.path.join('/mnt/pgth04b/DATABASES_CRIS/NO_FINALISTAS/DATABASES', ynf)
#    destino = os.path.join('/mnt/pgth04b/DATABASES_CRIS/solo_audio/no_finalistas', ynf)
#    dirs = os.listdir(folder_path)
#    for directorio in dirs:
#        path_inside_directorio = os.path.join(folder_path, directorio)
#        os.chdir(path_inside_directorio)
#        files = os.listdir(path_inside_directorio)
#        for f in files:
#            if f.endswith('.wav'):
#                dst = os.path.join(destino, f)
#                shutil.copy(f, dst)

# Extra copy

years_fin = ['2015', '2018']

for yf in years_fin:

    folder_path = os.path.join('/mnt/pgth04b/DATABASES_CRIS/FINALISTAS_ORIGINAL/DATABASES', yf)
    destino = os.path.join('/mnt/pgth04b/DATABASES_CRIS/solo_audio/finalistas', yf)
    dirs = os.listdir(folder_path)
    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        files = os.listdir(path_inside_directorio)
        for f in files:
            if f.endswith('.wav'):
                dst = os.path.join(destino, f)
                shutil.copy(f, dst)