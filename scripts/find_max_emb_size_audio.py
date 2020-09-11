import os
import numpy as np
import pandas as pd

years_f = ['2014', '2015', '2016', '2017', '2018', '2019']
path_f = '/mnt/pgth04b/DATABASES_CRIS/embeddings1024/finalistas'

years_nf = ['2017', '2018', '2019']
path_nf = '/mnt/pgth04b/DATABASES_CRIS/embeddings1024/no_finalistas'

def find_max(years, path):
    maxh = 0
    maxw = 0
    data = pd.read_csv('/mnt/pgth04b/DATABASES_CRIS/final_organization.csv')
    df = pd.DataFrame(data)
    corto = []
    for index, row in df.iterrows():
        if row['0 = CORTO | 1 = LARGO'] == 0:
            corto.append(row['id'])
    for y in years:
        path_to_saliency = os.path.join(path, y)
        saliency_in_year = os.listdir(path_to_saliency)
        for s in saliency_in_year:
            if s[0:11] in corto:
                path_audio = os.path.join(path_to_saliency, s)
                audio = np.load(path_audio)
                shape = np.shape(audio)
                if shape[1]>maxh:
                    maxh = shape[1]
                    path_long = path_audio

                if shape[2]>maxw:
                    maxw=shape[2]

    print('Shape is: Height--> ' + str(maxh) + ' Width --> ' + str(maxw))
    print(path_long)

find_max(years_f, path_f)
find_max(years_nf, path_nf)
