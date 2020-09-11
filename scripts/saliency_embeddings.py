#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

from keras.models import model_from_json
from keras.models import Model
from PIL import Image
from skimage import transform


def process_image(np_image):
    """
      Preprocess the image to be fed into the model.
    """
    np_image = np.array(np_image).astype('float32') / 255
    np_image = np_image.T
    np_image = transform.resize(np_image, (384, 224, 1))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def get_embeddings(path, years, finalista):
    model_path = '/home/aitorgalan/Escritorio/tfg-memorability-challenge-master/models/saliency/saliency_autoencoder_model.json'
    weight_path = '/home/aitorgalan/Escritorio/tfg-memorability-challenge-master/models/saliency/saliency_autoencoder_weight.h5'
    EMBEDDING_LENGTH = 84
    saliency_folder = 'output_tased'

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

    if finalista == 0:
        embeddings_folder = '/mnt/pgth04b/DATABASES_CRIS/embeddings_saliency/finalistas'
    else:
        embeddings_folder = '/mnt/pgth04b/DATABASES_CRIS/embeddings_saliency/no_finalistas'

    no_saliency_error = '/mnt/pgth04b/DATABASES_CRIS/videos_no_analizados_para_saliencia'
    list_of_errors = []
    count = 0

    for y in years:
        path_to_year = os.path.join(path, y)
        embeddings_folder_new = os.path.join(embeddings_folder, y)
        videos_in_year = os.listdir(path_to_year)
        videos_already_analyzed = os.listdir(embeddings_folder_new)
        for v in videos_in_year:
            count = count +1
            print(count)
            vcsv = v + '.csv'
            if vcsv not in videos_already_analyzed:
               
                video = os.path.join(path_to_year, v)
                folders_in_this_video = os.listdir(video)
                if saliency_folder not in folders_in_this_video:
                    list_of_errors.append(v)
                    

                else:
                    print('HABEMUS ANALYSIS')
                    saliency_frames = os.path.join(video, saliency_folder)
                    frames_in_video = os.listdir(saliency_frames)
                    data = []
                    for image in frames_in_video:
                        image_good = os.path.join(saliency_frames, image)
                        input_image = Image.open(image_good)
                        pixels = process_image(input_image)

                        # Get the embedding
                        embedding = intermediate_output_model.predict(pixels)

                        this_data = list(embedding[0])

                        # Save it in the array
                    this_data.insert(0, v)
                    this_data.insert(1, image)

                    data.append(this_data)

                    # Save data.
                    df = pd.DataFrame(data)
                    name_of_file = v + '.csv'
                    path_to_save_data = os.path.join(embeddings_folder_new, name_of_file)
                    df.to_csv(path_to_save_data)
                    print('Data saved for video ' + v)

    errorsdf = pd.DataFrame(list_of_errors)
    name_of_errors_file = 'videos_not_saliency_analisys.csv'
    saving_errors = os.path.join(no_saliency_error, name_of_errors_file)
    errorsdf.to_csv(saving_errors)



finalistas_saliency_path = '/mnt/pgth04b/DATABASES_CRIS/FINALISTAS_ORIGINAL/DATABASES'
no_finalistas_saliency_path = '/mnt/pgth04b/DATABASES_CRIS/NO_FINALISTAS/DATABASES'
years_finalistas = ['2014', '2015', '2016', '2017', '2018', '2019']
years_no_finalistas = ['2017', '2018', '2019']

get_embeddings(finalistas_saliency_path, years_finalistas,0)
get_embeddings(no_finalistas_saliency_path, years_no_finalistas,1)

