import numpy as np
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import metrics
import seaborn as sns

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

def cm_analysis(y_true, y_pred, filename, labels, labels_class, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    if labels_class == 'si_one_hot':
        y_true = y_true.argmax(axis=1)
        y_pred = y_pred.argmax(axis=1)
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm).to_csv(filename)


def lstm_audio():
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
    matrix_order = 0
    fig_order = 0
    kernel_regularizer = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)
    activity_regularizer = tf.keras.regularizers.l2(1e-5)

    train = []
    test = []

    label_train_sinonehot = []
    label_test_sinonehot = []

    max_length = 76
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
            finnofin_dir_listed = os.listdir(finnofin_dir)
            for v in finnofin_dir_listed:
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
    id_train = []
    for features, label in train:
        audio = np.load(features)
        shape_a = np.shape(audio)
        id_train.append(features[-14:-4])
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
    id_test = []
    for features, label in test:
        audio = np.load(features)
        id_test.append(features[-14:-4])
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

    #df = pd.DataFrame(data={"id_test": id_test})
    #df.to_csv("/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/final_audio_results/id_test.csv", sep=',', index=False)

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
    # LSTM neurons
    # activation
    # number of hidden LSTM layers

    c1 = [32, 0, 'softmax', 1]
    c2 = [32, 0, 'sigmoid', 1]
    c3 = [64, 0, 'sigmoid', 1]
    c4 = [64, 0, 'softmax', 1]
    c5 = [128, 0, 'sigmoid', 1]
    c6 = [128, 0, 'softmax', 1]
    c7 = [12, 32, 'softmax', 2]
    c8 = [12, 32, 'sigmoid', 2]
    c9 = [32, 64, 'sigmoid', 2]
    c10 = [32, 64, 'softmax', 2]
    c11 = [64, 64, 'softmax', 2]
    c12 = [64, 64, 'sigmoid', 2]
    c13 = [64, 128, 'sigmoid', 2]
    c14 = [64, 128, 'softmax', 2]

    # Combinations = [c1, c2, c3, c4]
    #combinations = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14]
    #combinations = [c1, c2, c3]
    #combinations = [c4, c5]
    #combinations = [c6, c7]
    #combinations = [c8, c9]
    #combinations = [c10, c11]
    #combinations = [c12, c13]
    combinations = [c7]
    whatc = 7
    for c in combinations:
        num_epochs = 100
        val_split = 0.2
        lstm_num1 = c[0]
        lstm_num2 = c[1]
        activation = c[2]

        if activation == 'softmax':
            labels_train = labels_train
            labels_test = labels_test
            labels_class = 'si_one_hot'
            dense_neurons = 2
        elif activation == 'sigmoid':
            labels_train = label_train_sinonehot
            labels_test = label_test_sinonehot
            dense_neurons = 1
            labels_class = 'no_one_hot'


        num_layers = c[3]
        loss = tf.keras.losses.BinaryCrossentropy()
        model = tf.keras.Sequential()
        optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(max_length, total_length)))


        if num_layers == 1:
            model.add(tf.keras.layers.LSTM(lstm_num1, input_shape=(max_length, total_length),
                                           activation='relu'))
        if num_layers == 2:
            model.add(tf.keras.layers.LSTM(lstm_num1, return_sequences=True, input_shape=(max_length, total_length),
                                           activation='relu'))
            model.add(tf.keras.layers.LSTM(lstm_num2, input_shape=(max_length, total_length)))
        if num_layers == 3:
            model.add(tf.keras.layers.LSTM(lstm_num, return_sequences=True, input_shape=(max_length, total_length),
                                           activation='relu'))
            model.add(tf.keras.layers.LSTM(lstm_num, return_sequences=True, input_shape=(max_length, total_length)))
            model.add(tf.keras.layers.LSTM(lstm_num, input_shape=(max_length, total_length)))

        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(dense_neurons, activation=activation, kernel_regularizer=kernel_regularizer,
                                            activity_regularizer=activity_regularizer))
        model.summary()

        ############# Esto se usará cuando metamos early stopping ################
        log_dir = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/models'
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch', write_images=True,write_graph=False)  # CONTROL THE TRAINING

        # CALLBACKS
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=2, mode='max', restore_best_weights=True)  # EARLY STOPPING
        ##########################################################################

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        #model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        history = model.fit(features_train, labels_train, epochs=num_epochs, validation_split=val_split, verbose=2,  callbacks=[tensorboard, earlyStopping])
        #history = model.fit(features_train, labels_train, epochs=num_epochs, validation_split=val_split, verbose=2)
        #history = model.fit(features_train, labels_train, epochs=num_epochs, validation_data=(features_test, labels_test), verbose=2)
        #history = model.fit(features_train, labels_train, epochs=num_epochs, validation_split=val_split)

        model_save_path = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/final_audio_results/models/comb' + str(whatc) + '_audio-lstm-model.json'
        weight_save_path = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/final_audio_results/models/comb' + str(whatc) + '_audio-lstm-weight.h5'

        with open(model_save_path, 'w+') as save_file:
            save_file.write(model.to_json())

        model.save_weights(weight_save_path)

        # Path for image
        img_file_path_loss = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/final_audio_results/figures/comb' + str(whatc) + '_audio_lstm_train_loss_class.png'
        img_file_path_acc = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/final_audio_results/figures/comb' + str(whatc) + '_audio_lstm_train_acc_class.png'

        #figure1_order += 1
        #figure2_order += 1
        #fig_order += 1
        #plot_graphs(history, "acc", img_file_path_acc, fig_order)
        #fig_order += 1
        #plot_graphs(history, "loss", img_file_path_loss, fig_order)

        fig1, ax1 = plt.subplots()
        ax1.plot(history.history['acc'])
        ax1.plot(history.history['val_acc'])
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel('acc')
        ax1.legend(['acc', 'val_acc'])
        fig1.savefig(img_file_path_acc)

        fig2, ax2 = plt.subplots()
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel('loss')
        ax2.legend(['loss', 'val_loss'])
        fig2.savefig(img_file_path_loss)

        # Predictions


        predictions = model.predict(features_test)
        predictions_train = model.predict(features_train)

        pred_path = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/final_audio_results/predictions/comb' + str(
            whatc) + '_predictions.csv'
        if dense_neurons ==2:
            predictions1 = []
            predictions2 = []
            for p in predictions:
                predictions1.append(p[0])
                predictions2.append(p[1])
            pd.DataFrame(data={"id_test": id_test, 'finalistas': predictions1, 'no_finalistas':predictions2}).to_csv(
                pred_path,index=False)
        else:
            predictions1 = []
            for p in predictions:
                predictions1.append(p[0])
            pd.DataFrame(data={"id_test": id_test, 'finalistas': predictions1}).to_csv(
                pred_path, index=False)


        pred_path_train= '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/final_audio_results/predictions/comb' + str(
            whatc) + '_predictions_train.csv'
        if dense_neurons == 2:
            predictions1_train = []
            predictions2_train = []
            for p in predictions_train:
                predictions1_train.append(p[0])
                predictions2_train.append(p[1])
            pd.DataFrame(data={"id_train": id_train, 'finalistas': predictions1_train, 'no_finalistas': predictions2_train}).to_csv(
                pred_path_train, index=False)
        else:
            predictions1_train = []
            for p in predictions_train:
                predictions1_train.append(p[0])
            pd.DataFrame(data={"id_train": id_train, 'finalistas': predictions1_train}).to_csv(
                pred_path_train, index=False)
        #res = pd.DataFrame(predictions)
        #res.index = features_test.index  # its important for comparison
        #res.columns = ["prediction"]
        #res.to_csv(pred_path)

        pred_path_05 = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/final_audio_results/predictions/comb' + str(
            whatc) + '_predictions_mayor05.csv'
        labels_predicted = (predictions > 0.5)
        pd.DataFrame(labels_predicted, id_test).to_csv(pred_path_05,index=False)

        mat_path = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/final_audio_results/conf_matrix/comb' + str(
            whatc) + '_confusion_matrix_in_csv.csv'
        name_labels = [0, 1]
        #cm_analysis(labels_test, labels_predicted, mat_path, name_labels, labels_class, ymap=None, figsize=(10, 10))
        y_true = labels_test
        y_pred = labels_predicted
        ymap = None
        if ymap is not None:
            y_pred = [ymap[yi] for yi in y_pred]
            y_true = [ymap[yi] for yi in y_true]
            labels = [ymap[yi] for yi in labels]
        if labels_class == 'si_one_hot':
            y_true = y_true.argmax(axis=1)
            y_pred = y_pred.argmax(axis=1)
        cm = metrics.confusion_matrix(y_true, y_pred, labels=name_labels)
        pd.DataFrame(cm).to_csv(mat_path, index=False)

        print('#######################################################################################################')
        print( 'That was combination ' + str(whatc))
        print('#######################################################################################################')
        whatc += 1

lstm_audio()
