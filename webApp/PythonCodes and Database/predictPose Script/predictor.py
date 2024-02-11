import pickle
import cv2
import mediapipe as mp
import numpy as np
import csv
import pandas as pd
import os

currDirect = os.path.dirname(os.path.realpath(__file__))
befDirect = os.path.dirname(currDirect)
mainDirect = os.path.dirname(befDirect)
modelPath = os.path.join(mainDirect, 'best_model.pkl')

file =  open(modelPath, "rb")

model = pickle.load(file)
a=model.steps[1][0]

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

clf = ["LogisticRegression","RandomForestClassifier","GradientBoostingClassifier"]
def giveClassifierName(c):
    s=""
    if(c.lower()==clf[0].lower()):
        s="Logistic Regression Model"
    elif(c.lower()==clf[1].lower()):
        s="Random Forest Model"
    else:
        s="Gradient Boosting Model"
    return s

frame_name = giveClassifierName(a)

def doPredictionsNew():
    cap = cv2.VideoCapture(0)
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1)
                                  )

        try:
            if results.pose_landmarks:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # MAKING DETECTIONS
                X = pd.DataFrame([pose_row])
                yoga_pose_name = model.predict(X)[0]
                yoga_pose_prob = model.predict_proba(X)[0]

                # Display yoga pose name and probability in frame on separate lines
                text1 = f"Yoga Pose: {yoga_pose_name}"
                text2 = f"Probability: {yoga_pose_prob.max():.2f}"
                cv2.rectangle(image, (10, 10), (600, 80), (0, 255, 255), -1)  # Yellow background
                cv2.putText(image, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black font
                cv2.putText(image, text2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red font

        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow("OPEN CV CAMERA", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
doPredictionsNew()
print("THANK YOU FOR USING MY YOGA POSE SYSTEM")