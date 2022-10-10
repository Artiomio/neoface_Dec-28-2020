import json
import glob
import math
from math import sin, cos
from time import time as current_time

import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
import keras


from fit_image import fit_img_center
def to_bw(img):
    if len(img.shape) == 2:
        return img
    R = img[:, :, 0].astype(np.long)
    G = img[:, :, 1].astype(np.long)
    B = img[:, :, 2].astype(np.long)
    img_new = ((R + G + B) / 3).astype(np.uint8)
    return img_new


model_path = glob.glob("model-22-11-2020.h5")[0] #'model100x100conv.model'
print(f"Loading model: {model_path}"); #input('Press Enter')
model = load_model(model_path, compile = False)


    





""" Finds the minimal horizontal rectangle containing given points
    points is a list of points e.g. [(3,5),(-1,4),(5,6)]
"""
def find_rect_range(points):
    min_x = min(points, key=lambda x: x[0])[0]
    max_x = max(points, key=lambda x: x[0])[0]

    min_y = min(points, key=lambda x: x[1])[1]
    max_y = max(points, key=lambda x: x[1])[1]
    return((min_x, min_y), (max_x, max_y))

def rotate(coords, origin, angle):
    """ Rotates given point around given origin
    """
    x, y = coords
    xc, yc = origin

    cos_angle = cos(angle)
    sin_angle = sin(angle)

    x_vector = x - xc
    y_vector = y - yc

    x_new = x_vector * cos_angle - y_vector * sin_angle + xc
    y_new = x_vector * sin_angle + y_vector * cos_angle + yc
    return (x_new, y_new)

from math import pi
def rotate_image(image, center, angle):
    row,col = image.shape[: 2]
    rot_mat = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image


def distance(x1,y1, x2, y2):
    return math.sqrt( (x1-x2)**2 + (y1-y2)**2)


def distance_N_dim_squared(a, b):
    return sum((np.array(a) - np.array(b)) ** 2)




def normalized_landmark_vector(landmarks):
    """ Нормализация "волшебных" точек
        На данный момент:
            Вертикальная ориентация лица
            Приведение к единичному масштабу
    """

    # Считаем угол таким образом, что положительное направление - склонённость к правому плечу
    # Центр - 28-я точка - т.е. landmarks[27]

    nose_bridge = landmarks[27]

    eyes_vector_x, eyes_vector_y = landmarks[45][0] - landmarks[36][0], landmarks[45][1] - landmarks[36][1]
    angle = - math.atan(eyes_vector_y / eyes_vector_x)
    
    
    #print("Угол равен %f градусов (наклон к правому плечу)" % (angle * 180 / math.pi))
    verticalized = [rotate((x,y), origin = nose_bridge, angle = angle) for (x, y) in landmarks]

    # Временно - как хеш лица используем только глаза
    # verticalized = verticalized[42:48] + verticalized[36:42]

    ((x1, y1), (x2, y2)) = find_rect_range(verticalized)
    width = x2 - x1
    height = y2 - y1
    
 
    normalized = verticalized
    normalized = [((x-x1) / width, (y-y1) / width) for (x, y) in verticalized]
    return normalized

def landmark_params(landmarks):
    landmarks = np.array(normalized_landmark_vector(landmarks))
    a = np.concatenate([ landmarks[:, 0].reshape(-1),
                         landmarks[:, 0].reshape(-1),
                         landmarks[:, 1].reshape(-1)])

    a = np.concatenate([a, np.flip(a)])
    return a
    

import dlib
predictor_path = r"/Users/art/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

face_rect = None
i = 0
while True:
    i += 1
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    if face_rect is not None:
        x, y, w, h = face_rect
        gray = gray[y - int(h * .1): y + int(h * 1.1) , x - int(w * .1) : x + int(1.1 * w)]
         
        img = gray
        
    
    if i % 100 == 0 or face_rect is None:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

      
        try:
            x, y, w, h = faces[0]
            img = gray
            face_rect = (x, y, w, h)
        except:
            face_rect = None

    
    faces, confidence, idx = detector.run(gray, 1) # Запускаем поиск лиц  
    
    if len(faces) < 1:
            cv2.putText(gray, "Can't see your face", (200, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
            face_rect = None
    else:
        
        face = faces[0]
        shape = predictor(gray, face) 
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(0, 68)])
        X = landmark_params(landmarks).reshape(1, -1)
        nn_img = (model.predict(X).reshape(120, 100) * 255).astype("uint8")

        nose_bridge = (int(landmarks[27][0]), int(landmarks[27][1]))                                                
        eyes_vector_x, eyes_vector_y = landmarks[45][0] - landmarks[36][0], landmarks[45][1] - landmarks[36][1]
        angle = math.atan(eyes_vector_y / eyes_vector_x)                           
                                                                                   
        vert_img = rotate_image(img, center=nose_bridge, angle=180 * angle / pi)

        nn_img = fit_img_center(nn_img, height=img.shape[0], width=img.shape[1])
        img = np.concatenate([nn_img, to_bw(vert_img)], axis=1)
        cv2.imshow('verticalized', np.flip(img, axis=1))

    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()
Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))