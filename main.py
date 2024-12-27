# import the necessary packages

from scipy.spatial import distance as dist
import imutils
from imutils.video import VideoStream
import numpy as np
import argparse
import time
import dlib
import cv2
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import streamlit as st

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

def app():
    prev_time = 0
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    face_detector = dlib.get_frontal_face_detector()
    keypoint_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#     image = cv2.imread('zara.png')
#     cap= cv2.VideoCapture(0)
    
    color = (255, 0, 0) 
    left_eye_points = range(36, 42)
    right_eye_points = range(42, 48)
    output = 'result1.mp4' 
    input_file = 'tommy.mp4'

    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 15
    COUNTER = 0
    FRAME_COUNTER = 0j
    EAR_COUNTER = 0
    EAR_LIST = []
    

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
          width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
          fps = int(cap.get(cv2.CAP_PROP_FPS))
          length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          fourcc = cv2.VideoWriter_fourcc(*'mp4v')
          writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

          ret, frame = cap.read()
         
          if not ret:
              break
        #  print("[INFO] processing frame ", {FRAME_COUNTER})
          FRAME_COUNTER+=1
          EAR_COUNTER+=1
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

          EAR_LIST.append(ear)
          EAR_COUNTER+=1
          if len(EAR_LIST)==30 and EAR_COUNTER ==30:
              EAR_LIST = []
              EAR_COUNTER=0

          if ear < EYE_AR_THRESH:
              COUNTER+=1

              if COUNTER >=EYE_AR_CONSEC_FRAMES:
                   cv2.putText(frame, f"detection alart", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

          else: 
              COUNTER=0

          # writer.write(frame)
          curr_time = time.time()
          fps = int(1/(curr_time - prev_time))
          prev_time = curr_time
          
          cv2.putText(frame,'FPS:' + str(fps),(50,70),cv2.FONT_HERSHEY_SIMPLEX,1,(149,255,149),2)
          # cv2.imshow("Face Landmarks", frame)
        
          key = cv2.waitKey(1)
          if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
#     writer.release()
  




if  __name__ == "__main__":
    app()

