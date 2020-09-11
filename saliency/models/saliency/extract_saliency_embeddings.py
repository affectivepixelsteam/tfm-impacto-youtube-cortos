import pandas as pd
import numpy as np
import os

from keras.models import model_from_json
from keras.models import Model
from PIL import Image
from skimage import transform

train_image_path = '/mnt/pgth06a/saliency/train/'
test_image_path = '/mnt/pgth06a/saliency/test/'
train_embeddings_output = '../../data/corpus/devset/dev-set/train_saliency_embeddings_splitted.csv'
test_embeddings_output = 'test_saliency_embeddings_splitted.csv'
model_path = 'saliency_autoencoder_model.json'
weight_path = 'saliency_autoencoder_weight.h5'
path_to_train = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/saliency/models/saliency/new_saliency/train'
path_to_test = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/saliency/models/saliency/new_saliency/test'

organize = pd.read_csv('/mnt/pgth04b/DATABASES_CRIS/final_organization.csv')
org_df = pd.DataFrame(organize)
corto = []
for index, row in org_df.iterrows():
    if row['0 = CORTO | 1 = LARGO'] == 0:
        corto.append(row['id'])

training = []
testing = []

for index, row in org_df.iterrows():
    if row['0 = TEST | 1 = TRAIN'] == 0:
        testing.append(row['id'])
    else:
        training.append(row['id'])

EMBEDDING_LENGTH = 84

def process_image(np_image):
    """
      Preprocess the image to be fed into the model.
    """
    np_image = np.array(np_image).astype('float32')/255
    np_image = np_image.T
    np_image = transform.resize(np_image, (384, 224, 1))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def final_process(video_folder, video_name, train_or_test):
    # load json and create model
    with open(model_path, 'r') as model_file:
        model = model_file.read()

    complete_model = model_from_json(model)

    # load weights into new model
    complete_model.load_weights(weight_path)
    print("Loaded model from disk")

    # Generate new model in which output is the bottleneck layer output so we can extract the embedding
    bottleneck_layer = "dense_1"
    intermediate_output_model = Model(
        inputs=complete_model.input, outputs=complete_model.get_layer(bottleneck_layer).output)

    # Create array in which we will save the embeddings
    data = []

    csv_name = video_name + '.csv'
    if train_or_test == 'train':
        csv_path = os.path.join(path_to_train, csv_name)
    else:
        csv_path = os.path.join(path_to_test, csv_name)

    images = os.listdir(video_folder)
    if len(images) != 0:
        print(video_name + ' is not empty!')
        for image in images:
            print(image)
            input_image = Image.open(os.path.join(video_folder, image))
            pixels = process_image(input_image)

            # Get the embedding
            embedding = intermediate_output_model.predict(pixels)

            this_data = list(embedding[0])

            # Save it in the array
            this_data.insert(0, video_name)
            this_data.insert(1, int(image[0:-4]))

            data.append(this_data)

        # Save data.
        df = pd.DataFrame(data)
        final_df = df.sort_values(1)
        final_df.to_csv(csv_path)

        print(video_name + ' was succesfully proccessed')
        final_df.head(2)


def execute_emb_saliency(years_f, path_f):

    output_tased = 'output_tased'

    for y in years_f:
        path_to_year = os.path.join(path_f, y)
        videos_in_year = os.listdir(path_to_year)

        for v in videos_in_year:
            if v in corto:
                video_name = v

                if v in testing:
                    train_or_test = 'test'
                else:
                    train_or_test = 'train'

                path_to_video = os.path.join(path_to_year, v)
                path_to_saliency = os.path.join(path_to_video, output_tased)
                final_process(path_to_saliency, video_name, train_or_test)

years_fin = ['2014', '2015', '2016', '2017', '2018', '2019']
path_fin = '/mnt/pgth04b/DATABASES_CRIS/FINALISTAS_ORIGINAL/DATABASES/'

years_no_fin = ['2017', '2018', '2019']
path_no_fin = '/mnt/pgth04b/DATABASES_CRIS/NO_FINALISTAS/DATABASES'

execute_emb_saliency(years_fin, path_fin)
execute_emb_saliency(years_no_fin, path_no_fin)
