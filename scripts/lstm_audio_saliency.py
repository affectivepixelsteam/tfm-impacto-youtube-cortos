import numpy as np
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

def plot_graphs(history, string,img_file_path, n):
    plt.figure(n)
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    if string == 'acc':
        plt.savefig(img_file_path)
    else:
        plt.savefig(img_file_path)

def lstm_a_s(category):
    data = pd.read_csv('/mnt/pgth04b/DATABASES_CRIS/final_organization.csv')
    df = pd.DataFrame(data)
    corto = []
    for index, row in df.iterrows():
        if row['0 = CORTO | 1 = LARGO'] == 0:
            corto.append(row['id'])

    train_test = ['train', 'test']
    label_options = ['finalistas', 'no_finalistas']

    onehot_finalitas = [1.0,0.0]
    onehot_nofinalistas = [0.0,1.0]

    figure1_order = 0
    figure2_order = 0

    kernel_regularizer = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)
    activity_regularizer = tf.keras.regularizers.l2(1e-5)
    if category == 'saliency':
        train = []
        test = []

        label_train_sinonehot = []
        label_test_sinonehot = []
        #max_length = 5000 #para largos
        #max_length = 900 #para cortos
        max_length = 76 #para 2 fps
        total_length = 84
        trunc_type = 'post'
        padding_type = 'post'
        help_test = 0
        help_train = 0
        #dir = '/mnt/pgth04b/DATABASES_CRIS/embeddings_train_test_division/saliency'
        dir = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/saliency/models/saliency/short_saliency'
        for t in train_test:
            traintest_dir = os.path.join(dir, t)
            for l in label_options:
                class_num = label_options.index(l)
                finnofin_dir = os.path.join(traintest_dir,l)

                for v in os.listdir(finnofin_dir):
                    #if v[0:11] in corto:
                        path = os.path.join(finnofin_dir, v)
                        if t == 'train':
                            train.append([path, class_num])
                        else:
                            test.append([path, class_num])

        random.shuffle(train)
        random.shuffle(test)

        features_train = []
        features_test = []
        labels_train = []
        labels_test = []

        for features, label in train:

            saliency = pd.read_csv(features, index_col=0, thousands=',')
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
                saliency_padded = pad_sequences(salarr, maxlen=max_length, padding=padding_type,
                                                truncating=trunc_type)

                if help_train == 0:
                    help_train = 1
                    features_train = saliency_padded
                else:
                    features_train = np.concatenate([features_train, saliency_padded], axis=0)

                if label == 0:
                    label_train_sinonehot.append(0)
                    labels_train.append(onehot_finalitas)
                else:
                    label_train_sinonehot.append(1)
                    labels_train.append(onehot_nofinalistas)

        for features, label in test:

            saliency = pd.read_csv(features, index_col=0, thousands=',')
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
                saliency_padded = pad_sequences(salarr, maxlen=max_length, padding=padding_type,
                                                truncating=trunc_type)

                if help_test == 0:
                    help_test = 1
                    features_test = saliency_padded
                else:
                    features_test = np.concatenate([features_test, saliency_padded], axis=0)

                if label == 0:
                    label_test_sinonehot.append(0)
                    labels_test.append(onehot_finalitas)
                else:
                    label_test_sinonehot.append(1)
                    labels_test.append(onehot_nofinalistas)

        labels_train = np.array(labels_train)
        labels_test = np.array(labels_test)
        label_test_sinonehot = np.array(label_test_sinonehot)
        label_train_sinonehot = np.array(label_train_sinonehot)

        print('shape train is:')
        print(np.shape(features_train))
        print(len(labels_train))
        print('**********************')
        print('shape test is:')
        print(np.shape(features_test))
        print(len(labels_test))
        print('**********************')



        # Parameters order in combinations:
        # learning rate
        # epochs
        # split
        # lstm_num
        # activation
        # labels
        # dense neurons

        c1 = [0.0001, 20, 0.2, 32, 'softmax', 'si_one_hot', 2]
        c2 = [0.0001, 20, 0.2, 32, 'sigmoid', 'no_one_hot',1]
        c3 = [0.0001, 200, 0.3, 64, 'sigmoid']
        c4 = [0.0001, 200, 0.3, 64, 'softmax']
        #combinations = [c1, c2, c3, c4]
        combinations = [c1, c2]
        for c in combinations:
            whatc = combinations.index(c)
            learning_rate = c[0]
            num_epochs = c[1]
            val_split = c[2]
            lstm_num = c[3]
            activation = c[4]
            dense_neurons = c[6]
            if c[5] == 'no_one_hot':
                labels = label_train_sinonehot
            elif c[5] == 'si_one_hot':
                labels = labels_train

            model = tf.keras.Sequential()
            optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
            model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(max_length, total_length)))
            model.add(tf.keras.layers.LSTM(lstm_num, return_sequences=True, input_shape=(max_length, total_length),
                                           activation='relu'))

            ############################### Para cuando funcione con return_sequences #########################
            model.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(max_length, total_length)))
            model.add(tf.keras.layers.LSTM(64, input_shape=(max_length, total_length)))
            ###################################################################################################

            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(dense_neurons, activation=activation,kernel_regularizer=kernel_regularizer,
        activity_regularizer=activity_regularizer))
            model.summary()

            ############# Esto se usará cuando metamos early stopping ################
            # log_dir = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/models'
            # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch', write_images=True,
            # write_graph=False)  # CONTROL THE TRAINING

            # CALLBACKS
            # earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto',
            # restore_best_weights=True)  # EARLY STOPPING
            ##########################################################################

            model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
            #model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

            ##history = model.fit(embeddings_train, labels_train, epochs=num_epochs, validation_split=val_split,
            # verbose=2, callbacks=[tensorboard, earlyStopping])
            #history = model.fit(features_train, labels_train, epochs=num_epochs, validation_split=val_split, verbose=2)
            #history = model.fit(features_train, labels_train, epochs=num_epochs, validation_data=(features_test, labels_test), verbose=2)
            history = model.fit(features_train, labels, epochs=num_epochs, validation_split=val_split, verbose=2)

            model_save_path = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/results_lstm_audio_saliency/models/saliency/comb' + str(whatc) + '_saliency-lstm-model.json'
            weight_save_path = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/results_lstm_audio_saliency/models/saliency/comb' + str(whatc) + '_saliency-lstm-weight.h5'

            with open(model_save_path, 'w+') as save_file:
                save_file.write(model.to_json())

            model.save_weights(weight_save_path)

            # Path for image
            img_file_path_loss = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/results_lstm_audio_saliency/figures/saliency/comb' + str(whatc) + '_saliency_lstm_train_loss_class.png'
            img_file_path_acc = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/results_lstm_audio_saliency/figures/saliency/comb' + str(whatc) + '_saliency_lstm_train_acc_class.png'

            figure1_order += 1
            figure2_order += 1
            plot_graphs(history, "acc",img_file_path_acc, figure1_order)
            plot_graphs(history, "loss",img_file_path_loss, figure2_order)

    elif category == 'audio':
        train = []
        test = []

        label_train_sinonehot = []
        label_test_sinonehot = []

        max_length = 443
        total_length = 1024
        trunc_type = 'post'
        padding_type = 'post'
        help_test = 0
        help_train = 0
        dir = '/mnt/pgth04b/DATABASES_CRIS/embeddings_train_test_division/audio'
        for t in train_test:
            traintest_dir = os.path.join(dir, t)
            for l in label_options:
                class_num = label_options.index(l)
                finnofin_dir = os.path.join(traintest_dir, l)

                for v in os.listdir(finnofin_dir):
                    if v[0:11] in corto:
                        path = os.path.join(finnofin_dir, v)

                        if t == 'train':
                            train.append([path, class_num])
                        else:
                            test.append([path, class_num])

        random.shuffle(train)
        random.shuffle(test)

        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for features, label in train:
            audio = np.load(features)
            shape_a = np.shape(audio)
            audio_padded = pad_sequences(audio, maxlen=max_length, padding=padding_type, truncating=trunc_type)
            if help_train == 0:
                help_train = 1
                features_train = audio_padded
            else:
                features_train = np.concatenate([features_train, audio_padded], axis=0)

            if label == 0:
                label_train_sinonehot.append(0)
                labels_train.append(onehot_finalitas)
            else:
                label_train_sinonehot.append(1)
                labels_train.append(onehot_nofinalistas)

        for features, label in test:
            audio = np.load(features)
            shape_a = np.shape(audio)
            audio_padded = pad_sequences(audio, maxlen=max_length, padding=padding_type, truncating=trunc_type)
            if help_test == 0:
                help_test = 1
                features_test = audio_padded
            else:
                features_test = np.concatenate([features_test, audio_padded], axis=0)

            if label == 0:
                label_test_sinonehot.append(0)
                labels_test.append(onehot_finalitas)
            else:
                label_test_sinonehot.append(1)
                labels_test.append(onehot_nofinalistas)

        labels_train = np.array(labels_train)
        labels_test = np.array(labels_test)
        label_test_sinonehot = np.array(label_test_sinonehot)
        label_train_sinonehot = np.array(label_train_sinonehot)

        print('shape train is:')
        print(np.shape(features_train))
        print(len(labels_train))
        print('**********************')
        print('shape test is:')
        print(np.shape(features_test))
        print(len(labels_test))
        print('**********************')

        # Parameters order in combinations:
        # learning rate
        # epochs
        # split
        # lstm_num
        # activation
        # labels
        # dense neurons

        c1 = [0.0001, 20, 0.2, 32, 'softmax', 'si_one_hot', 2]
        c2 = [0.0001, 20, 0.2, 32, 'sigmoid', 'no_one_hot', 1]
        c3 = [0.0001, 200, 0.3, 64, 'sigmoid']
        c4 = [0.0001, 200, 0.3, 64, 'softmax']
        # combinations = [c1, c2, c3, c4]
        combinations = [c1, c2]
        for c in combinations:
            whatc = combinations.index(c)
            learning_rate = c[0]
            num_epochs = c[1]
            val_split = c[2]
            lstm_num = c[3]
            activation = c[4]
            labels = c[5]
            dense_neurons = c[6]
            if labels == 'no_one_hot':
                labels = label_train_sinonehot
            elif labels == 'si_one_hot':
                labels = labels_train

            model = tf.keras.Sequential()
            optimizer = tf.keras.optimizers.Adam()
            model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(max_length, total_length)))
            model.add(tf.keras.layers.LSTM(lstm_num, return_sequences=True, input_shape=(max_length, total_length),
                                           activation='relu'))

            ############################### Para cuando funcione con return_sequences #########################
            model.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(max_length, total_length)))
            model.add(tf.keras.layers.LSTM(64, input_shape=(max_length, total_length)))
            ###################################################################################################

            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(dense_neurons, activation=activation, kernel_regularizer=kernel_regularizer,
                                            activity_regularizer=activity_regularizer))
            model.summary()

            ############# Esto se usará cuando metamos early stopping ################
            # log_dir = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/models'
            # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch', write_images=True,
            # write_graph=False)  # CONTROL THE TRAINING

            # CALLBACKS
            # earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto',
            # restore_best_weights=True)  # EARLY STOPPING
            ##########################################################################

            model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
            #model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

            ##history = model.fit(embeddings_train, labels_train, epochs=num_epochs, validation_split=val_split,
            # verbose=2, callbacks=[tensorboard, earlyStopping])
            #history = model.fit(features_train, labels_train, epochs=num_epochs, validation_split=val_split, verbose=2)
            #history = model.fit(features_train, labels_train, epochs=num_epochs, validation_data=(features_test, labels_test), verbose=2)
            history = model.fit(features_train, labels, epochs=num_epochs, validation_split=val_split)

            model_save_path = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/results_lstm_audio_saliency/models/audio/comb' + str(whatc) + '_audio-lstm-model.json'
            weight_save_path = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/results_lstm_audio_saliency/models/audio/comb' + str(whatc) + '_audio-lstm-weight.h5'

            with open(model_save_path, 'w+') as save_file:
                save_file.write(model.to_json())

            model.save_weights(weight_save_path)

            # Path for image
            img_file_path_loss = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/results_lstm_audio_saliency/figures/audio/comb' + str(whatc) + '_audio_lstm_train_loss_class.png'
            img_file_path_acc = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/results_lstm_audio_saliency/figures/audio/comb' + str(whatc) + '_audio_lstm_train_acc_class.png'

            figure1_order += 1
            figure2_order += 1
            plot_graphs(history, "acc", img_file_path_acc, figure1_order)
            plot_graphs(history, "loss", img_file_path_loss, figure2_order)



lstm_a_s('saliency')
#lstm_a_s('audio')


