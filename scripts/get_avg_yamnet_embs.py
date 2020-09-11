import numpy as np
import os
import pandas as pd

def get_avg_embs(embs2compact_path, axis=1):
    """
    Extract mean of passed embedding
    :param embs2compact: emb to get the average in columns.
    """
    emb = np.load(embs2compact_path, allow_pickle=True)
    avg_emb = np.nanmean(emb, axis=axis)
    avg_emb = avg_emb.reshape(-1)
    return avg_emb



if __name__ == "__main__":
    partitions = ["train", "test"]
    labels = ["finalistas", "no_finalistas"]
    root_path_embs = "/mnt/pgth04b/DATABASES_CRIS/embeddings_train_test_division/audio"
    output_path_embs_avg = "/mnt/pgth04b/DATABASES_CRIS/embeddings_train_test_division/audio_avg"
    output_path_df = "/mnt/pgth04b/DATABASES_CRIS/embeddings_train_test_division/csv"
    extract_avg_embs = False
    generate_csv = True
    path_distribution_short_audios = '/mnt/pgth04b/DATABASES_CRIS/final_organization.csv'

    if(extract_avg_embs):
        for partition in partitions:
            for label in labels:
                #input folder
                root_input_path_embs = os.path.join(root_path_embs, partition, label)
                #output folder
                root_output_path_embs = os.path.join(output_path_embs_avg, partition, label)
                os.makedirs(root_output_path_embs, exist_ok=True)
                for video_name in os.listdir(root_input_path_embs):
                    emb_path = os.path.join(root_input_path_embs, video_name)
                    #Extract average
                    avg_emb = get_avg_embs(emb_path)
                    #save emb
                    out_emb_path = os.path.join(root_output_path_embs, video_name.split(".")[0])
                    np.save(out_emb_path,avg_emb)

    if(generate_csv):
        distribution_df = pd.read_csv(path_distribution_short_audios)
        os.makedirs(output_path_df, exist_ok=True)
        #0-FINALISTA / 1-NO_FINALISTA
        #0-CORTO / 1-LARGO
        #0 = TEST | 1 = TRAIN
        distribution_df.columns = ['id', 'year', 'label',
       'video_length', 'partition', 'View_Count',
       'Like_Count', 'Dislike_Count', 'Duration', 'Title', 'fps',
       'Frames_Width', 'Frames_Height', 'Num_Comments', 'Upload_date',
       'Age_Limit', 'Categories', 'Description', 'Average_Rating',
       'Playlist_Index', 'Tags', 'Score']
        #Extract only short videos:
        short_videos_df = distribution_df.loc[distribution_df["video_length"]==0]
        #Extract training samples from short videos
        df_train_videos = short_videos_df.loc[short_videos_df["partition"]==1]
        # Extract test samples from short videos
        df_test_videos = short_videos_df.loc[short_videos_df["partition"] == 0]
        #Fill in datasets with train embs
        df_train = pd.DataFrame([], columns=list(range(0,1024))+["label"]+["video_name"])
        df_test = pd.DataFrame([], columns=list(range(0, 1024))+["label"]+["video_name"])
        print("Generating train csv ...")
        for index, train_video_row in df_train_videos.iterrows():
            video_id = train_video_row["id"]
            video_label = train_video_row["label"]
            label_name = "finalistas" if(video_label==0) else "no_finalistas"
            input_path_emb = os.path.join(root_path_embs, "train", label_name, video_id+".npy")
            #load emb:
            if(os.path.isfile(input_path_emb)):
                emb = np.load(input_path_emb)
            else:
                print("NOT FOUND: ", video_id)
            df_train = df_train.append(pd.DataFrame([list(emb)+[video_label]+[video_id]], columns=list(range(0,1024))+["label"]+["video_name"]))
        #Save train dataframe:
        df_train.to_csv(os.path.join(output_path_df, "train_avg_audio.csv"), sep=",", index=False, header=True)
        # Fill in datasets with test embs
        print("Generating test csv ...")
        for index, test_video_row in df_test_videos.iterrows():
            video_id = test_video_row["id"]
            video_label = test_video_row["label"]
            label_name = "finalistas" if(video_label==0) else "no_finalistas"
            input_path_emb = os.path.join(root_path_embs, "test", label_name,video_id+".npy")
            #load emb:
            if (os.path.isfile(input_path_emb)):
                emb = np.load(input_path_emb)
            else:
                print("NOT FOUND: ", video_id)
            df_test = df_test.append(pd.DataFrame([list(emb)+[video_label]+[video_id]], columns=list(range(0,1024))+["label"]+["video_name"]))
        #Save train dataframe:
        df_test.to_csv(os.path.join(output_path_df, "test_avg_audio.csv"), sep=",", index=False, header=True)
