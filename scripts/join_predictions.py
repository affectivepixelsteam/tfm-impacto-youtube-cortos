import pandas as pd
import arff


def join_predictions(prediction_video_weka, prediction_audio_weka, prediction_audio_lstm, filename_out):
    total_arff_video = '/mnt/pgth04b/DATABASES_CRIS/FEATURES_ARFF/video_test.arff'
    df_audio = pd.read_csv(prediction_audio_lstm)
    df_video_weka = pd.read_csv(prediction_video_weka)
    df_audio_weka = pd.read_csv(prediction_audio_weka)
    data_arff_video = arff.load(open(total_arff_video, 'r'))['data']
    ids = []
    for n in range(len(data_arff_video)):
        id_to_append = data_arff_video[n][-2][1:]
        ids.append(id_to_append)

    df_video_weka['id_test'] = ids
    df_video_weka.sort_values(by=['id_test'])
    actual_video = df_video_weka['actual'].tolist()
    predicted_video = df_video_weka['predicted'].tolist()
    prob_video = df_video_weka['prediction'].tolist()
    error_video = df_video_weka['error'].tolist()

    df_audio_weka['id_test'] = ids
    df_audio_weka.sort_values(by=['id_test'])
    actual_audio_weka = df_audio_weka['actual'].tolist()
    predicted_audio_weka = df_audio_weka['predicted'].tolist()
    prob_audio_weka = df_audio_weka['prediction'].tolist()
    error_audio_weka = df_audio_weka['error'].tolist()

    df_audio.sort_values(by=['id_test'])
    audio_shape = df_audio.shape
    predicted_audio = []
    value_audio = []

    for index, row in df_audio.iterrows():
        if row['id_test'] not in ids:
            df_audio.drop(index, inplace=True)

    if audio_shape[1] == 3:
        for index, row in df_audio.iterrows():
            fnf = [row['finalistas'],row['no_finalistas']]
            value_audio.append(max(fnf))
            if max(fnf) == row['finalistas']:
                predicted_audio.append('1:0.0')
            else:
                predicted_audio.append('2:1.0')
    else:
        for index, row in df_audio.iterrows():
            value_audio.append(row['finalistas'])
            if row['finalistas'] > 0.5:
                predicted_audio.append('1:0.0')
            else:
                predicted_audio.append('2:1.0')

            if row['id_test'] not in ids:
                df_audio.drop(index)

    error_audio = [0] * len(data_arff_video)
    for p in range(len(data_arff_video)):
        if predicted_audio[p] != actual_video[p]:
            error_audio[p] = '+'


    output = {'actual class': actual_video, 'predicted video weka': predicted_video,
              'prob video weka': prob_video, 'predicted audio weka': predicted_audio_weka,
              'prob audio weka': prob_audio_weka, 'predicted audio lstm': predicted_audio,
              'prob audio lstm': value_audio}
    output_df = pd.DataFrame(data=output)
    output_df.to_csv(filename_out, index=False)



video_weka = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/prueba_join/predictions_on_test_video.csv'
audio_weka = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/prueba_join/predictions_on_test_audio.csv'
audio_lstm = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/final_audio_results/predictions/comb6_predictions.csv'
output = '/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/prueba_join/prediction_combined.csv'
join_predictions(video_weka, audio_weka, audio_lstm, output)

