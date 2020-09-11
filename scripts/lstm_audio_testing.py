import numpy as np
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Hyperparameters
number_of_audios = 1510
embedding_dim = 1024
# Let's create the lists for training and testing

data = pd.read_csv('/mnt/pgth04b/DATABASES_CRIS/final_organization.csv')
df = pd.DataFrame(data)
training_audios = []
testing_audios = []

for index, row in df.iterrows():
    if row['0 = TEST | 1 = TRAIN'] == 0:
        testing_audios.append(row['id'])
    else:
        training_audios.append(row['id'])


 # first we set the lists for embeddings and labels
#embeddings_train = []
labels_train = []
#embeddings_test = []
labels_test = []

# FINALISTAS = 0

years_f = ['2014', '2015', '2016', '2017', '2018', '2019']
path_f = '/mnt/pgth04b/DATABASES_CRIS/embeddings1024/finalistas'

look_for_max = []
max_length = 443
trunc_type = 'post'
padding_type = 'post'
help_test = 0
help_train = 0
for y in years_f:
    path_to_audio = os.path.join(path_f,y)
    audios_in_year = os.listdir(path_to_audio)
    for a in audios_in_year:
        path = os.path.join(path_to_audio,a)
        audio = np.load(path)
        print(audio)
        break
        shape_a = np.shape(audio)
        audio_padded = pad_sequences(audio, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        look_for_max.append(shape_a[1])
        if a[0:11] in testing_audios:
            if help_test == 0:
                help_test = 1
                embeddings_test = audio_padded
            else:
                embeddings_test = np.concatenate([embeddings_test, audio_padded], axis=0)
            labels_test.append(0)
        else:
            if help_train == 0:
                help_train = 1
                embeddings_train = audio_padded
            else:
                embeddings_train = np.concatenate([embeddings_train, audio_padded], axis=0)
            labels_train.append(0)

# NO FINALISTAS = 1

years_nf = ['2017', '2018', '2019']
path_nf = '/mnt/pgth04b/DATABASES_CRIS/embeddings1024/no_finalistas'
for y in years_nf:
    path_to_audio = os.path.join(path_nf,y)
    audios_in_year = os.listdir(path_to_audio)
    for a in audios_in_year:
        path = os.path.join(path_to_audio, a)
        audio = np.load(path)
        shape_a = np.shape(audio)
        audio_padded = pad_sequences(audio, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        look_for_max.append(shape_a[1])
        if a[0:11] in testing_audios:
            embeddings_test = np.concatenate([embeddings_test, audio_padded], axis=0)
            labels_test.append(1)
        else:
            embeddings_train = np.concatenate([embeddings_train, audio_padded], axis=0)
            labels_train.append(1)

print('shape train is:')
print(np.shape(embeddings_train))
print(len(labels_train))
print('**********************')
print('shape test is:')
print(np.shape(embeddings_test))
print(len(labels_test))
print('**********************')


# Now the chicha chicha
#model = tf.keras.Sequential()
#model.add(tf.keras.layers.LSTM(32, input_shape=(443, 1024)))
#model.add(tf.keras.layers.Dense(2, activation='softmax'))
#model.summary()
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#num_epochs = 10
#history = model.fit(embeddings_train, labels_train, epochs=num_epochs, validation_data=(embeddings_test, labels_test),verbose=2)


# Path model and weights
#model_save_path = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/models/audio/audio-lstm-model.json'
#weight_save_path = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/models/audio/audio-lstm-weight.h5'


#with open(model_save_path, 'w+') as save_file:
#    save_file.write(model.to_json())

#model.save_weights(weight_save_path)

# Path for image
#img_file_path_loss = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/figures/audio/audio_lstm_train_loss_class.png'
#img_file_path_acc = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/figures/audio/audio_lstm_train_acc_class.png'

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    if string == 'acc':
        plt.savefig(img_file_path_acc)
    else:
        plt.savefig(img_file_path_loss)
        
    plt.show()


#plot_graphs(history, "acc")
#plot_graphs(history, "loss")

########################################################################################################################
########################################################################################################################
########################################################################################################################

