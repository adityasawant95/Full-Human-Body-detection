# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:44:10 2021

@author: aditya
"""

import mediapipe as mp
import cv2 #OpenCV for camera 

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_drawing.DrawingSpec(color = (255,255,0),thickness = 1.5, circle_radius=1.5)
#%%
#Get Realitime Webcam Feed
cap = cv2.VideoCapture(0)
#Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:

    #cap = cv2.VideoCapture(2)
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Flip Image to get mirror effect
        image = cv2.flip(image,1)
        
        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #1_Draw face landmark
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (255,255,0),thickness = 2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color = (255,255,0),thickness = 2, circle_radius=2))
        
        #2_Draw Left hand landmartk
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color = (150,255,0),thickness = 2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color = (150,255,0),thickness = 2, circle_radius=2))
        
        #3_Draw Rigth hand landmark
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (150,255,0),thickness = 2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color = (150,255,0),thickness = 2, circle_radius=2))
        
        #4_Draw Pose detection landmark
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (255,0,255),thickness = 2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color = (255,0,255),thickness = 2, circle_radius=2))
        
        cv2.imshow('Raw webcam Feed',image)
        
        if  cv2.waitKey(10) & 0xFF == ord('q'):
            break

#%%
cap.release()
cv2.destroyAllWindows()