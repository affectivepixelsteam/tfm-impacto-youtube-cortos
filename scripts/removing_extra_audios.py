import shutil
import os
import pandas as pd

years_no_fin = ['2017', '2018', '2019']
data = pd.read_csv('/mnt/pgth04b/DATABASES_CRIS/final_organization.csv')
df = pd.DataFrame(data)
audios_to_remove = []
audios_removed = []
new_data = df.loc[df['0 = FINALISTA | 1 = NO FINALISTA'] == 1, ['id']]
dlist = new_data['id'].tolist()
print(dlist)
print(len(dlist))

for ynf in years_no_fin:
    folder_path = os.path.join('/mnt/pgth04b/DATABASES_CRIS/solo_audio/no_finalistas', ynf)
    audios = os.listdir(folder_path)

    for a in audios:
        file = os.path.join(folder_path, a)
        id = a[0:11]
        if id not in dlist:
            audios_to_remove.append(a)
            os.remove(file)
            print(a + ' was succesfully removed!')
            audios_removed.append(a)

print(audios_to_remove)
print(len(audios_to_remove))
