import cv2
import streamlit as st
import numpy as np
from scipy.spatial import distance as dist
import imutils
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import time
import dlib
from main import eye_aspect_ratio
import pandas as pd



def update_chart(chart_data):
    st.line_chart(chart_data)



st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

prev_time = 0
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
face_detector = dlib.get_frontal_face_detector()
keypoint_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

color = (255, 0, 0) 
left_eye_points = range(36, 42)
right_eye_points = range(42, 48)
output = 'result1.mp4' 
input_file = 'tommy.mp4'

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 15
COUNTER = 0
FRAME_COUNTER = 0
EAR_COUNTER = 0
EAR_LIST = []
prev_data = []
old_data = pd.DataFrame(prev_data, columns=['Eye Aspect Ratio'])
chart = st.line_chart(old_data)

while run:
    ret, frame = camera.read()
    if not ret:
        break
    print("[INFO] processing frame ", {FRAME_COUNTER})
    FRAME_COUNTER+=1
   
    left_eye_POI = []
    right_eye_POI = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces)==0:
        continue
    for face in faces:       
        face_landmarks = keypoint_predictor(gray, face)
        for n in range(0, 67):    
            if n in left_eye_points:       
                    left_eye_POI.append((face_landmarks.part(n).x, face_landmarks.part(n).y ))

            elif n in right_eye_points:
                right_eye_POI.append((face_landmarks.part(n).x, face_landmarks.part(n).y ))

            if n in left_eye_points or n in right_eye_points:
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    cv2.putText(frame, str(n), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, color, 1, cv2.LINE_AA)          
                
    left_EAR = eye_aspect_ratio(left_eye_POI)
    right_EAR = eye_aspect_ratio(right_eye_POI)   

    ear = (left_EAR+right_EAR)/2.0

    EAR_LIST.append(round(ear, 2))
    EAR_COUNTER+=1
    

    new_data = pd.DataFrame(EAR_LIST, columns=['Eye Aspect Ratio'])
    chart_data = pd.concat([old_data, new_data])
    chart.line_chart(new_data)
    prev_data=EAR_LIST

    EAR_COUNTER=0
    old_data = pd.DataFrame(EAR_LIST, columns=['Eye Aspect Ratio'])


    if ear < EYE_AR_THRESH:
        COUNTER+=1

        if COUNTER >=EYE_AR_CONSEC_FRAMES:
            cv2.putText(frame, f"detection alart", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    else: 
        COUNTER=0

    curr_time = time.time()
    fps = int(1/(curr_time - prev_time))
    prev_time = curr_time
    
    cv2.putText(frame,'FPS:' + str(fps),(50,70),cv2.FONT_HERSHEY_SIMPLEX,1,(149,255,149),2)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()



