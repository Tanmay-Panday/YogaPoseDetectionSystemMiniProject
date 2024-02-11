import csv
import os
import numpy as np
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic     

def CusCam():
    cap = cv2.VideoCapture(0)
    holistic =mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    while cap.isOpened():
        ret, frame = cap.read() #ret = true <if camera is available> frame = image array vector
    
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1)
                                 )
        cv2.imshow("OPEN CV CAMERA", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return results

def check_coordinates_file(path):
    return os.path.exists(path)

def createDatabase(path_OP):
    results = CusCam()
    total_cords = len(results.pose_landmarks.landmark)

    landmarks = ['class']
    for val in range(1,total_cords+1):
        landmarks = landmarks + ['x{}'.format(val), "y{}".format(val), "z{}".format(val), "v{}".format(val)]

    with open(path_OP, mode = 'w',newline='') as file:
        csv_editor = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_editor.writerow(landmarks)

def createDatabaseIFNotExisting():
    script_directory = os.path.dirname(os.path.realpath(__file__))
    PythonCodesAndDatabase = os.path.dirname(script_directory)
    coordinates_path = os.path.join(PythonCodesAndDatabase, 'Database', 'coordinates.csv')

    # coordinates_path has path to coordinates.csv in DatabaseTrial folder
    if check_coordinates_file(coordinates_path):
        print("Database Already exists")
    else :
        createDatabase(coordinates_path)
        print("Database created")

createDatabaseIFNotExisting()