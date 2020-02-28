import pandas as pd
import numpy as np

csv_file = '/mnt/pgth04b/DATABASES_CRIS/FINALISTAS_vs_NOFINALISTAS.csv'

data = pd.read_csv(csv_file)
df = pd.DataFrame(data)

video_id = []
like_rating = []
view_norm = []

duration = []

def norm_view(year, views):
    if year == 2014:
        divide = 6
    elif year == 2015:
        divide = 5
    elif year == 2016:
        divide = 4
    elif year == 2017:
        divide = 3
    elif year == 2018:
        divide = 2
    else:
        divide = 1
    view_norm = views/divide
    return view_norm


for index, row in df.iterrows():
    if row['label(0=FINALISTAS/1=NO_FINALISTAS)'] == 1:
        video_id.append(row['id'])
        likes = row['like_count']
        dislike = row['dislike_count']
        lidis = likes + dislike
        if lidis == 0:
            lidis = 1
        like_rating.append(likes/lidis)
        year = row['festival_year']
        view = row['view_count']
        view_norm.append(norm_view(year, view))
        if row['duration'] > 40:
            duration.append(1)
        else:
            duration.append(0)

df_to_order = pd.DataFrame({'id': video_id,
                            'likes_rating': like_rating,
                            'view_norm': view_norm,
                            '1 = LARGO | 0 = CORTO': duration})

sorted_view_df = df_to_order.sort_values(['view_norm'])

sorted_likes_df = df_to_order.sort_values(['likes_rating'])


video_view = []
position_view = []
video_likes = []
position_likes = []

for index, row in sorted_view_df.iterrows():
    video_view.append(row['id'])
    position_view.append(index)

view_index = pd.DataFrame({'id':video_view,
                           'view_index':position_view})


for index, row in sorted_likes_df.iterrows():
    video_likes.append(row['id'])
    position_likes.append(index)

likes_index = pd.DataFrame({'id':video_likes,
                           'likes_index':position_likes})

total_position = []

for index, row in df_to_order.iterrows():

    for index_v, row_v in view_index.iterrows():
        if row['id'] == row_v['id']:
            pos_v = row_v['view_index']
            print('pos_v is ' + str(pos_v))

    for index_l, row_l in likes_index.iterrows():
        if row['id'] == row_l['id']:
            pos_l = row_l['likes_index']
            print('pos_l is ' + str(pos_l))
    total_position.append((pos_l+pos_v)/2)

print(total_position)

final_df = pd.DataFrame({'id': video_id,
                         'total rating':total_position})

print(final_df)

final_sorted = final_df.sort_values(['total rating'])

final_sorted.to_csv('/mnt/pgth04b/DATABASES_CRIS/no_finalistas_ordenados.csv', index = False)

print(final_sorted)
