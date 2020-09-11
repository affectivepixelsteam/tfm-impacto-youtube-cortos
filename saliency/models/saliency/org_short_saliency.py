import os
import pandas as pd
import shutil

dir = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/saliency/models/saliency/short_saliency'

tt = ['train', 'test']
fnf = ['finalistas', 'no_finalistas']

f = []
nf = []
data = pd.read_csv('/mnt/pgth04b/DATABASES_CRIS/final_organization.csv')
df = pd.DataFrame(data)
for index, row in df.iterrows():
    if row['0 = FINALISTA | 1 = NO FINALISTA'] == 0:
        f.append(row['id'])
    else:
        nf.append(row['id'])

for t in tt:

    path = os.path.join(dir, t)
    move_to_fin = os.path.join(path, fnf[0])
    move_to_no_fin = os.path.join(path, fnf[1])

    videos = os.listdir(path)

    for v in videos:
        video_path = os.path.join(path, v)
        if v[0:11] in f:
            shutil.move(video_path, move_to_fin)

        elif v[0:11] in nf:
            shutil.move(video_path, move_to_no_fin)

test_fin = os.listdir('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/saliency/models/saliency/short_saliency/test/finalistas')
print('number of test finalistas ' + str(len(test_fin)))
test_nofin = os.listdir('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/saliency/models/saliency/short_saliency/test/no_finalistas')
print('number of test no finalistas ' + str(len(test_nofin)))
train_fin = os.listdir('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/saliency/models/saliency/short_saliency/train/finalistas')
print('number of train finalistas ' + str(len(train_fin)))
train_nofin = os.listdir('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/saliency/models/saliency/short_saliency/train/no_finalistas')
print('number of train no finalistas ' + str(len(train_nofin)))
