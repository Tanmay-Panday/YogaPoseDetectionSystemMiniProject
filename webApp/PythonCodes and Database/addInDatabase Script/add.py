import os
import cv2
import mediapipe as mp
import numpy as np
import time
import csv

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

currDirect = os.path.dirname(os.path.realpath(__file__))
mainDirect = os.path.dirname(currDirect)
DatabasePath= os.path.join(mainDirect, 'Database', 'coordinates.csv')

def givePoseCoords(poseName):
    cap = cv2.VideoCapture(0)
    if cap:
        print("camera accessible\n")
    else:
        print("camera inaccessible\n")

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Set the timer duration
    timer_duration = 10
    
    # Get the current time to start the timer
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()  # ret = true <if camera is available> frame = image array vector

        # Calculate the elapsed time
        elapsed_time = time.time() - start_time
        remaining_time = max(0, timer_duration - elapsed_time)
        
        # Draw the countdown timer on the frame
        cv2.putText(frame, f"Timer: {int(remaining_time)} seconds", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

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
               pose_row.insert(0, poseName)
               with open(DatabasePath, mode='a', newline='') as f:
                   csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                   csv_writer.writerow(pose_row)
        except Exception as e:
            print(f"Error: {e}")


        cv2.imshow("OPEN CV CAMERA", image)

        if elapsed_time >= timer_duration:
            break

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    f.close()
    cap.release()
    cv2.destroyAllWindows()

givePoseCoords(input())
print("Yoga pose inserted")