import sys
import os
import cv2
import csv
from keras.models import load_model

import numpy as np
import tensorflow as tf

from oarriaga.src.utils.datasets import get_labels
from oarriaga.src.utils.inference import detect_faces
from oarriaga.src.utils.inference import draw_text
from oarriaga.src.utils.inference import draw_bounding_box
from oarriaga.src.utils.inference import apply_offsets
from oarriaga.src.utils.inference import load_detection_model
from oarriaga.src.utils.inference import load_image
from oarriaga.src.utils.preprocessor import preprocess_input

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# sess = tf.Session(config=config)
# print(tf.test.is_gpu_available)

# parameters for loading data and images
#image_path = sys.argv[1]
#image_path = '/mnt/RESOURCES/tfm-impacto-youtube-cortos/DATABASES/2014/825D-GUiSEU/825D-GUiSEU_frames/825D-GUiSEU481.jpg'
ROOT_PATH = "/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/oarriaga"
detection_model_path = 'trained_models/detection_models/haarcascade_frontalface_default.xml'
detection_model_path = os.path.join(ROOT_PATH, detection_model_path)
emotion_model_path = 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_model_path = os.path.join(ROOT_PATH, emotion_model_path)
gender_model_path = 'trained_models/gender_models/simple_CNN.81-0.96.hdf5'
gender_model_path = os.path.join(ROOT_PATH, gender_model_path)
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
gender_offsets = (10, 10)
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

 x =
# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_classifier.summary()
gender_classifier = load_model(gender_model_path, compile=False)
gender_classifier.summary()

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

years = ['2014', '2015', '2016', '2017', '2018']


for y in years:

    folder_path = os.path.join('/mnt/RESOURCES/tfm-impacto-youtube-cortos/DATABASES', y)
    dirs = os.listdir(folder_path)

    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        frames_folder = directorio + '_frames'
        path_to_frames = os.path.join(path_inside_directorio, frames_folder)
        frames = os.listdir(path_to_frames)
        video_id = directorio
        csv_headers = [['frame_id', 'face_id', 'man', 'woman', 'emotion', 'max_attention', 'thirds rule', 'bounding box position']]
        csv_name = 'faces_information.csv'
        csv_path = os.path.join(path_inside_directorio, csv_name)
        name_new_dir = frames_folder + '_analyzed'
        os.mkdir(name_new_dir)
        path_to_frames_analyzed = os.path.join(path_inside_directorio, name_new_dir)

        for frame in frames:
            img_path = os.path.join(path_to_frames, frame)
            frame_id = frame[:-4]
            # loading images
            rgb_image = load_image(img_path, 'rgb', grayscale=False)
            gray_image = load_image(img_path, 'grayscale', grayscale=True)
            gray_image = np.squeeze(gray_image)
            gray_image = gray_image.astype('uint8')

            # Coordinadas de los focos de máxima atencion

            img_frame = cv2.imread(img_path, 0)
            height, width = img_frame.shape[:2]
            height_short = int(height * (1 / 3))
            height_long = int(height * (2 / 3))
            width_short = int(width * (1 / 3))
            width_long = int(width * (2 / 3))
            height = int(height)
            width = int(width)
            top_left = (width_short, height_short)
            bottom_left = (width_short, height_long)
            top_right = (width_long, height_short)
            bottom_right = (width_long, height_long)
            size_frame = width*height
            distance_horizontal_lines = height_long - height_short
            distance_vertical_lines = width_long - width_short

            # Dibujar focos y lineas
            cv2.line(rgb_image, (0, height_short), (width, height_short), 1, 3)
            cv2.line(rgb_image, (0, height_long), (width, height_long), 1, 3)
            cv2.line(rgb_image, (width_short, 0), (width_short, height), 1, 3)
            cv2.line(rgb_image, (width_long, 0), (width_long, height), 1, 3)
            cv2.circle(rgb_image, (width_short, height_short), 5, (255, 0, 0))
            cv2.circle(rgb_image, (width_short, height_long), 5, (255, 0, 0))
            cv2.circle(rgb_image, (width_long, height_short), 5, (255, 0, 0))
            cv2.circle(rgb_image, (width_long, height_long), 5, (255, 0, 0))


            faces = detect_faces(face_detection, gray_image)
            for index, face_coordinates in enumerate(faces):

                face_id = frame_id + '_' + index

                x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
                rgb_face = rgb_image[y1:y2, x1:x2]

                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]

                # Finding the position of the face
                corner_x1 = face_coordinates.item(0)
                corner_x2 = face_coordinates.item(1)
                size_y1 = face_coordinates.item(2)
                size_y2 = face_coordinates.item(3)

                center_of_bounding_box = (corner_x1 + int(size_y1/2), corner_x2 + int(size_y2/2))
                height_bb = size_y1
                width_bb = size_y2
                size_bounding_box = size_y1*size_y2
                relation_box_frame = size_bounding_box/size_frame
                corner_left_small_box = int(center_of_bounding_box.__getitem__(0) - distance_horizontal_lines/6)
                corner_right_small_box = int(center_of_bounding_box.__getitem__(0) + distance_horizontal_lines/6)
                corner_up_small_box = int(center_of_bounding_box.__getitem__(1) - distance_vertical_lines/6)
                corner_down_small_box = int(center_of_bounding_box.__getitem__(1) + distance_vertical_lines/6)

                face_centered_focus = '-1'

                # Comprobar si está centrada en un foco de máxima atención

                if (top_left.__getitem__(0) in range(corner_left_small_box, corner_right_small_box)) and (
                        top_left.__getitem__(1) in range(corner_up_small_box, corner_down_small_box)):
                    face_centered_focus = 'Centrada en foco superior izquierda'
                elif (top_right.__getitem__(0) in range(corner_left_small_box, corner_right_small_box)) and (
                        top_right.__getitem__(1) in range(corner_up_small_box, corner_down_small_box)):
                    face_centered_focus = 'Centrada en foco superior derecha'
                elif (bottom_right.__getitem__(0) in range(corner_left_small_box, corner_right_small_box)) and (
                        bottom_right.__getitem__(1) in range(corner_up_small_box, corner_down_small_box)):
                    face_centered_focus = 'Centrada en foco inferior derecha'
                elif (bottom_left.__getitem__(0) in range(corner_left_small_box, corner_right_small_box)) and (
                        bottom_left.__getitem__(1) in range(corner_up_small_box, corner_down_small_box)):
                    face_centered_focus = 'Centrada en foco inferior izquierda'

                bounding_box_position = '[ Esquina izquierda en (' + str(corner_x1) + ' ,' + str(corner_x2) + ') y tamaño ('  + str(size_y1) + ' ,' +str(size_y2) + ') ]'

                # Regla de los tercios

                face_in = '-1'

                if center_of_bounding_box.__getitem__(0) < width_short:
                    if height_bb > height_short:
                        face_in = 'tercio 1'
                    elif center_of_bounding_box.__getitem__(1) < height_short:
                        face_in = 'noveno 1'
                    elif center_of_bounding_box.__getitem__(1) < height_long:
                        face_in = 'noveno 2'
                    elif center_of_bounding_box.__getitem__(1) > height_short:
                        face_in = 'noveno 3'
                elif center_of_bounding_box.__getitem__(0) < width_long:
                    if height_bb > height_short:
                        face_in = 'tercio 2'
                    elif center_of_bounding_box.__getitem__(1) < height_short:
                        face_in = 'noveno 4'
                    elif center_of_bounding_box.__getitem__(1) < height_long:
                        face_in = 'noveno 5'
                    elif center_of_bounding_box.__getitem__(1) > height_short:
                        face_in = 'noveno 6'
                elif center_of_bounding_box.__getitem__(0) > width_long:
                    if height_bb > height_short:
                        face_in = 'tercio 3'
                    elif center_of_bounding_box.__getitem__(1) < height_short:
                        face_in = 'noveno 7'
                    elif center_of_bounding_box.__getitem__(1) < height_long:
                        face_in = 'noveno 8'
                    elif center_of_bounding_box.__getitem__(1) > height_short:
                        face_in = 'noveno 9'

                try:
                    rgb_face = cv2.resize(rgb_face, (gender_target_size))
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_percentage = emotion_classifier.predict(gray_face)
                emotion_label_arg = np.argmax(emotion_percentage)
                emotion_text = emotion_labels[emotion_label_arg]

                emotion = {'angry': emotion_percentage.item(0)*100,
                           'disgust': emotion_percentage.item(1)*100,
                           'fear': emotion_percentage.item(2)*100,
                           'happy': emotion_percentage.item(3)*100,
                           'sad': emotion_percentage.item(4)*100,
                           'surprise': emotion_percentage.item(5)*100,
                           'neutral': emotion_percentage.item(6)*100}

                rgb_face = preprocess_input(rgb_face, False)
                rgb_face = np.expand_dims(rgb_face, 0)
                gender_prediction = gender_classifier.predict(rgb_face)
                gender_label_arg = np.argmax(gender_prediction)
                gender_text = gender_labels[gender_label_arg]

                gender = {'woman': gender_prediction.item(0)*100,
                          'man': gender_prediction.item(1)*100}

                data = [frame_id, face_id, gender, emotion, face_centered_focus, face_in, bounding_box_position]

                with open(csv_name, 'w+') as csv_file_end:
                    writer = csv.writer(csv_file_end)
                    writer.writerows(data)
                csv_file_end.close()

                if gender_text == gender_labels[0]:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)
                draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            an_frame = frame_id + '_analyzed.jpg'
            path_to_new_image = os.path.join(path_to_frames_analyzed, an_frame)
            cv2.imwrite(path_to_new_image, bgr_image)