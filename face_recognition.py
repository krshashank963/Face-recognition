# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:14:09 2020

@author: SHASHANK RAJPUT
"""


import dlib
import scipy.misc
import numpy as np
import os

face_detector = dlib.get_frontal_face_detector()

shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat_2')

face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat_2')

TOLERANCE = 0.55


def face_encode(path):
   
    image = scipy.misc.imread(path)
    
    detected_faces = face_detector(image, 1)
   
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]


def compare(known_faces, face):
 
    return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE)


def find_match(known_faces, names, face):
   
    matches = compare(known_faces, face)
   
    count = 0
    for match in matches:
        if match:
            return names[count]
        count += 1

image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('images/'))

image_filenames = sorted(image_filenames)

paths_to_images = ['images/' + x for x in image_filenames]

face_encodings = []

for path in paths_to_images:
   
    face_encode_image = face_encode(path)
    
    if len(face_encode_image) != 1:
        print("Please change image: " + path + " - it has " + str(len(face_encode_image)) + " faces; it can only have one")
        exit()

    face_encodings.append(face_encode(path)[0])
    
    
test_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('test/'))
paths_to_test_images = ['test/' + x for x in test_filenames]

names = [x[:-4] for x in image_filenames]

for path in paths_to_test_images:
   
    face_encode_image = face_encode(path)
    
    if len(face_encode_image) != 1:
        print("Please change image: " + path +  " faces; it can only have one")
        exit()
    
    match = find_match(face_encodings, names, face_encode_image[0])
    
    print(path, match)    

