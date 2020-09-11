import pandas as pd
import os
import numpy as np


years_f = ['2014', '2015', '2016', '2017', '2018', '2019']
path_f = '/mnt/pgth04b/DATABASES_CRIS/embeddings_saliency/finalistas'

years_nf = ['2017', '2018', '2019']
path_nf = '/mnt/pgth04b/DATABASES_CRIS/embeddings1024/no_finalistas'

def find_max(years, path):
    maxh = 0
    maxw = 0
    notempty = 0
    for y in years:
        path_to_saliency = os.path.join(path, y)
        saliency_in_year = os.listdir(path_to_saliency)
        for s in saliency_in_year:
            if s[-3:] != 'npy':
                path_df = os.path.join(path_to_saliency, s)
                saliency = pd.read_csv(path_df, index_col=0, thousands=',')
                if not saliency.empty:
                    notempty += 1
                    print(str(notempty))
                    saliency = saliency.drop(['0', '1'], axis=1)
                    salarr = saliency.to_numpy()
                    sh = salarr.shape[0]
                    sw = salarr.shape[1]
                    if sh > maxh:
                        maxh = salarr.shape[0]
                    if sw > maxw:
                        maxw = salarr.shape[1]

    size = [maxh, maxw]
    return size

max_f = find_max(years_f, path_f)
max_nf = find_max(years_nf, path_nf)

if max_f[0] > max_nf[0]:
    print('Max h size is ' + str(max_f[0]))
else:
    print('Max h size is ' + str(max_nf[0]))

if max_f[1] > max_nf[1]:
    print('Max w size is ' + str(max_f[1]))
else:
    print('Max w size is ' + str(max_nf[1]))
