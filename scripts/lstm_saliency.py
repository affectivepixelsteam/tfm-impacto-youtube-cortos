import numpy as np
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences




data = pd.read_csv('/mnt/pgth04b/DATABASES_CRIS/final_organization.csv')
df = pd.DataFrame(data)
training_saliency = []
testing_saliency = []

for index, row in df.iterrows():
    if row['0 = TEST | 1 = TRAIN'] == 0:
        testing_saliency.append(row['id'])
    else:
        training_saliency.append(row['id'])


 # first we set the lists for embeddings and labels
#embeddings_train = []
labels_train = []
#embeddings_test = []
labels_test = []


appendix0 = [1.0,0.0]
appendix1 = [0.0,1.0]

# FINALISTAS = 0

years_f = ['2014', '2015', '2016', '2017', '2018', '2019']
path_f = '/mnt/pgth04b/DATABASES_CRIS/embeddings_saliency/finalistas'

look_for_max = []
max_length = 5000
total_length = 88
trunc_type = 'post'
padding_type = 'post'
help_test = 0
help_train = 0
for y in years_f:
    path_to_saliency = os.path.join(path_f,y)
    saliency_in_year = os.listdir(path_to_saliency)
    for a in saliency_in_year:
        path = os.path.join(path_to_saliency,a)
        saliency = pd.read_csv(path, index_col = 0, thousands  = ',')
        if not saliency.empty:
            saliency = saliency.drop(['0','1'], axis=1)
            salarray = saliency.to_numpy()
            for i, l in enumerate(salarray):
                for n, h in enumerate(salarray[i]):
                    if isinstance(salarray[i, n], str):
                        salarray[i] = 0
            shape_array = np.shape(salarray)
            primero = shape_array[0]
            if shape_array[1] >= total_length:
                columns = range(total_length, shape_array[1])
                salarray2 = np.delete(salarray, columns , axis=1)
            else:
                segundo = total_length - shape_array[1]
                ceros = np.zeros((primero, segundo), dtype=int)
                salarray2 = np.concatenate([salarray, ceros], axis=1)
            shape_a = np.shape(salarray2)
            salarr = salarray2.reshape((1, shape_a[0], shape_a[1]))
            saliency_padded = pad_sequences(salarr, maxlen=max_length, padding=padding_type, truncating=trunc_type)
            look_for_max.append(shape_a[1])
            if a[0:11] in testing_saliency:
                if help_test == 0:
                    help_test = 1
                    embeddings_test = saliency_padded
                else:
                    embeddings_test = np.concatenate([embeddings_test, saliency_padded], axis=0)
                labels_test.append(appendix0)
            else:
                if help_train == 0:
                    help_train = 1
                    embeddings_train = saliency_padded
                else:
                    embeddings_train = np.concatenate([embeddings_train, saliency_padded], axis=0)
                labels_train.append(appendix0)

print('Finalistaas Done')
# NO FINALISTAS = 1

years_nf = ['2017', '2018', '2019']
path_nf = '/mnt/pgth04b/DATABASES_CRIS/embeddings_saliency/no_finalistas'
for y in years_nf:
    path_to_saliency = os.path.join(path_nf,y)
    saliency_in_year = os.listdir(path_to_saliency)
    for a in saliency_in_year:
        path = os.path.join(path_to_saliency, a)
        saliency = pd.read_csv(path, index_col=0, thousands=',', encoding= 'unicode_escape')
        if not saliency.empty:
            saliency = saliency.drop(['0', '1'], axis=1)
            salarray = saliency.to_numpy()
            for i, l in enumerate(salarray):
                for n, h in enumerate(salarray[i]):
                    if isinstance(salarray[i, n], str):
                        salarray[i] = 0
            shape_array = np.shape(salarray)
            primero = shape_array[0]
            if shape_array[1] >= total_length:
                columns = range(total_length, shape_array[1])
                salarray2 = np.delete(salarray, columns, axis=1)
            else:
                segundo = total_length - shape_array[1]
                ceros = np.zeros((primero, segundo), dtype=int)
                salarray2 = np.concatenate([salarray, ceros], axis=1)
            shape_a = np.shape(salarray2)
            salarr = salarray2.reshape((1, shape_a[0], shape_a[1]))
            saliency_padded = pad_sequences(salarr, maxlen=max_length, padding=padding_type, truncating=trunc_type)
            look_for_max.append(shape_a[1])
            if a[0:11] in testing_saliency:
                embeddings_test = np.concatenate([embeddings_test, saliency_padded], axis=0)
                labels_test.append(appendix1)
            else:
                embeddings_train = np.concatenate([embeddings_train, saliency_padded], axis=0)
                labels_train.append(appendix1)

labels_train = np.array(labels_train)
labels_test = np.array(labels_test)

print('shape train is:')
print(np.shape(embeddings_train))
print(len(labels_train))
print('**********************')
print('shape test is:')
print(np.shape(embeddings_test))
print(len(labels_test))
print('**********************')

# Now the chicha chicha
model = tf.keras.Sequential()
optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.add(tf.keras.layers.Masking(mask_value=0., input_shape=(max_length, total_length)))
model.add(tf.keras.layers.LSTM(32, input_shape=(max_length, total_length), activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.summary()

log_dir = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/models'
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch', write_images=True,
                                             write_graph=False)  # CONTROL THE TRAINING

# CALLBACKS
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  # val_loss
                                              patience=10, verbose=2, mode='auto',
                                              restore_best_weights=True)  # EARLY STOPPING


model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
num_epochs = 40
val_split = 0.2
history = model.fit(embeddings_train, labels_train, epochs=num_epochs, validation_split=val_split,
                    verbose=2, callbacks=[tensorboard, earlyStopping])


# Path model and weights
model_save_path = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/models/saliency/saliency-lstm-model.json'
weight_save_path = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/models/saliency/saliency-lstm-weight.h5'


with open(model_save_path, 'w+') as save_file:
    save_file.write(model.to_json())

model.save_weights(weight_save_path)

# Path for image
img_file_path_loss = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/figures/saliency/saliency_lstm_train_loss_class.png'
img_file_path_acc = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/figures/saliency/saliency_lstm_train_acc_class.png'

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


plot_graphs(history, "acc")
plot_graphs(history, "loss")

########################################################################################################################
########################################################################################################################
########################################################################################################################
