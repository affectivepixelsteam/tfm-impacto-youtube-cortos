import pandas as pd

csv_file = pd.read_csv('/mnt/pgth04b/DATABASES_CRIS/data_merged.csv')
data = pd.DataFrame(csv_file)
data = data.iloc[:,0:20]
data = data.sort_values(['festival_year'])

bad_videos = pd.read_csv('/mnt/pgth04b/DATABASES_CRIS/no_finalistas_ordenados.csv')
bad_data = pd.DataFrame(bad_videos)

finalistas_cortos = []
no_finalistas_cortos = []

finalistas_largos = []
no_finalistas_largos = []

header = ['id', 'year', '1 = LARGO | 0 = CORTO', '0 = FINALISTA | 1 = NO FINALISTA']

for index, row in data.iterrows():

    if row['label(0=FINALISTAS/1=NO_FINALISTAS)'] == '0':
        if row['duration'] > 40:
            finalistas_largos.append(row['id'])
        else:
            finalistas_cortos.append(row['id'])

num_largos = len(finalistas_largos)
num_cortos = len(finalistas_cortos)

label_finalistas_largos = ['1']*num_largos
label_finalistas_cortos = ['0']*num_cortos


for index, row in bad_data.iterrows():
    id = row['id']
    for i, r in data.iterrows():
        if r['id'] == id:
            if r['duration'] > 40:
                if len(no_finalistas_largos) < num_largos:
                    no_finalistas_largos.append(id)
            else:
                if len(no_finalistas_cortos) < num_cortos:
                    no_finalistas_cortos.append(id)

num_no_cortos = len(no_finalistas_cortos)
num_no_largos = len(no_finalistas_largos)
label_no_finalistas_largos = ['1']*num_no_largos
label_no_finalistas_cortos = ['0']*num_no_cortos


train_finalistas_cortos = int(0.8*len(finalistas_cortos))
test_finalistas_cortos = len(finalistas_cortos)-train_finalistas_cortos
train_test_finalistas_cortos = ['0']*test_finalistas_cortos + ['1']*train_finalistas_cortos

train_finalistas_largos = int(0.8*len(finalistas_largos))
test_finalistas_largos = len(finalistas_largos)-train_finalistas_largos
train_test_finalistas_largos = ['0']*test_finalistas_largos + ['1']*train_finalistas_largos

train_no_finalistas_cortos = int(0.8*len(no_finalistas_cortos))
test_no_finalistas_cortos = len(no_finalistas_cortos)-train_no_finalistas_cortos
train_test_no_finalistas_cortos = ['0']*test_no_finalistas_cortos + ['1']*train_no_finalistas_cortos

train_no_finalistas_largos = int(0.8*len(no_finalistas_largos))
test_no_finalistas_largos = len(no_finalistas_largos)-train_no_finalistas_largos
train_test_no_finalistas_largos = ['0']*test_no_finalistas_largos + ['1']*train_no_finalistas_largos

train_test = train_test_finalistas_cortos + train_test_finalistas_largos + train_test_no_finalistas_cortos + \
             train_test_no_finalistas_largos

videos = finalistas_cortos + finalistas_largos + no_finalistas_cortos + no_finalistas_largos

label_corto_largo = label_finalistas_cortos + label_finalistas_largos + label_no_finalistas_cortos + \
                    label_no_finalistas_largos

label_finalistas_largos = ['0']*num_largos
label_finalistas_cortos = ['0']*num_cortos
label_no_finalistas_largos = ['1']*num_no_largos
label_no_finalistas_cortos = ['1'
                              '']*num_no_cortos

label_fin_no_fin = label_finalistas_cortos + label_finalistas_largos + label_no_finalistas_cortos + \
                   label_no_finalistas_largos

new_year =[]
new_age_limit = []
new_average_rating = []
new_categories = []
new_description = []
new_dislike_count = []
new_like_count = []
new_view_count = []
new_duration = []
new_fps = []
new_frames_height = []
new_frames_width = []
new_fulltitle = []
new_num_comments = []
new_playlist_index = []
new_score = []
new_tags = []
new_upload_date = []
check = []
for name in videos:
    for index, row in data.iterrows():
        if row['id'] == name and name not in check:
            new_year.append(row['festival_year'])
            new_age_limit.append(row['age_limit'])
            new_average_rating.append(row['average_rating'])
            new_categories.append(row['categories'])
            new_description.append(row['description'])
            new_dislike_count.append(row['dislike_count'])
            new_like_count.append(row['like_count'])
            new_view_count.append(row['view_count'])
            new_duration.append(row['duration'])
            new_fps.append(row['fps'])
            new_frames_height.append(row['frames_heigth'])
            new_frames_width.append(row['frames_width'])
            new_fulltitle.append(row['fulltitle'])
            new_num_comments.append(row['num_comments'])
            new_playlist_index.append(row['playlist_index'])
            new_upload_date.append(row['upload_date'])
            new_tags.append(row['tags'])
            new_score.append(row['score'])
    check.append(name)

final_df = pd.DataFrame({'id': videos,
                         'year': new_year,
                         '0 = FINALISTA | 1 = NO FINALISTA': label_fin_no_fin,
                         '0 = CORTO | 1 = LARGO': label_corto_largo,
                         '0 = TEST | 1 = TRAIN': train_test,
                         'View Count': new_view_count,
                         'Like Count': new_like_count,
                         'Dislike Count': new_dislike_count,
                         'Duration': new_duration,
                         'Title': new_fulltitle,
                         'fps': new_fps,
                         'Frames Width': new_frames_width,
                         'Frames Height': new_frames_height,
                         'NUm Comments': new_num_comments,
                         'Upload date': new_upload_date,
                         'Age Limit': new_age_limit,
                         'Categories': new_categories,
                         'Description': new_description,
                         'Average Rating': new_average_rating,
                         'Playlist Index': new_playlist_index,
                         'Tags': new_tags,
                         'Score': new_score})

print(final_df)

df = final_df.sort_values(by=['0 = CORTO | 1 = LARGO', '0 = FINALISTA | 1 = NO FINALISTA', 'year',
                              '0 = TEST | 1 = TRAIN'])

df.to_csv('/mnt/pgth04b/DATABASES_CRIS/final_organization.csv', index=False)

