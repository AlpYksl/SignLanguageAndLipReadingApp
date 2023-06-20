import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import csv
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import math
mp_holistic = mp.solutions.holistic # Holistic german_german_german_model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
def detectFeatures(image, results, draw=True, display=True):
    # Get the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image to write the count of fingers on.
    output_image = image.copy()
    
    # Initialize a dictionary to store the count of fingers of both hands.
    count = {'RIGHT': 0, 'LEFT': 0}
    
    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [mp_holistic.HandLandmark.INDEX_FINGER_TIP, mp_holistic.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_holistic.HandLandmark.RING_FINGER_TIP, mp_holistic.HandLandmark.PINKY_TIP,
                        mp_holistic.HandLandmark.THUMB_TIP]
    
    # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}
    
    # Initialize a list to store the landmarks of the finger tips of each finger that is up for both hands.
    fingers_landmarks = {'RIGHT': [], 'LEFT': []}
    
    # Get the landmarks of both hands.
    right_hand_landmarks = results.right_hand_landmarks
    left_hand_landmarks = results.left_hand_landmarks
    
    # Iterate over both hands.
    for hand_landmarks, hand_label in [(left_hand_landmarks, 'LEFT'), (right_hand_landmarks, 'RIGHT')]:
        if hand_landmarks is not None:
            # Iterate over the indexes of the tips landmarks of each finger of the hand.
            # Iterate over the indexes of the tips landmarks of each finger of the hand.
            for tip_index in fingers_tips_ids:
                
                # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
                finger_name = tip_index.name.split("_")[0]
                
                # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
                if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                    
                    # Update the status of the finger in the dictionary to true.
                    fingers_statuses[hand_label.upper()+"_"+finger_name] = True
                    
                    # Increment the count of the fingers up of the hand by 1.
                    count[hand_label.upper()] += 1
                    
                    # Add the landmark of the finger tip to the list of landmarks of the finger tips of the hand.
                    fingers_landmarks[hand_label.upper()].append(hand_landmarks.landmark[tip_index])
            
            # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
            thumb_tip_x = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP].x
            thumb_mcp_x = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP - 2].x
        
        
            # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
    if (hand_label=='Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label=='Left' and (thumb_tip_x > thumb_mcp_x)):
        
        # Update the status of the thumb in the dictionary to true.
        fingers_statuses[hand_label.upper()+"_THUMB"] = True
        
        # Increment the count of the fingers up of the hand by 1.
        count[hand_label.upper()] += 1
        
    shoulder_landmark = []
    distance = 0
   
    if results.pose_landmarks and hand_landmarks:
        if results.left_hand_landmarks:
            
            shoulder_landmark = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            
        else:
            
            shoulder_landmark = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            
        # Get the x, y, z-coordinates of the hand and the shoulder.
        hand_x, hand_y, hand_z = hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x, hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y, hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].z
        shoulder_x, shoulder_y, shoulder_z = shoulder_landmark.x, shoulder_landmark.y, shoulder_landmark.z
        # Calculate the distance between the hand and the shoulder.
        distance = math.sqrt((hand_x - shoulder_x) ** 2 + (hand_y - shoulder_y) ** 2 + (hand_z - shoulder_z) ** 2)
        # Draw the distance between the hand and the shoulder on the output image.
        if draw:    
            cv2.putText(output_image, f"Distance: {distance:.2f}", (int(shoulder_landmark.x*width), int(shoulder_landmark.y*height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            
    
            
    angle_between_fingers = 0
    # Calculate the angle between the index finger and the thumb, if both fingers are up.
    if fingers_statuses[hand_label.upper()+"_INDEX"] and fingers_statuses[hand_label.upper()+"_THUMB"]:
        # Get the landmarks of the index finger and the thumb.
        index_finger_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
        wrist = hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
        
        # Calculate the angle between the index finger and the thumb using the dot product.
        thumb_to_index_finger_vector = np.array([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y, thumb_tip.z - wrist.z])
        index_finger_tip_to_wrist_vector = np.array([index_finger_tip.x - wrist.x, index_finger_tip.y - wrist.y, index_finger_tip.z - wrist.z])
        
        if np.linalg.norm(thumb_to_index_finger_vector) == 0 or np.linalg.norm(index_finger_tip_to_wrist_vector) == 0:
            angle_between_fingers = 0 # ya da başka bir özel değer ataması yapabilirsiniz.
        else:
            angle_between_fingers = np.arccos(np.dot(thumb_to_index_finger_vector, index_finger_tip_to_wrist_vector) / (np.linalg.norm(thumb_to_index_finger_vector) * np.linalg.norm(index_finger_tip_to_wrist_vector)))
            angle_between_fingers = np.degrees(angle_between_fingers)
     
        # Draw the angle between the index finger and the thumb on the output image.
        if draw:
            cv2.putText(output_image, f"{angle_between_fingers:.2f}°", (int(thumb_tip.x*width), int(thumb_tip.y*height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
   
    # Check if the total count of the fingers of both hands are specified to be written on the output image.
    if draw:
        # Write the total count of the fingers of both hands on the output image.
        cv2.putText(output_image, " Total Fingers: ", (10, 25),cv2.FONT_HERSHEY_COMPLEX, 1, (20,255,155), 2)
        cv2.putText(output_image, str(sum(count.values())), (width//2-150,240), cv2.FONT_HERSHEY_SIMPLEX,
                    8.9, (20,255,155), 10, 10)

    # Check if the output image is specified to be displayed.
    if display:
        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    # Otherwise
    else:
        for key in fingers_statuses:
            if fingers_statuses[key] == True:
                fingers_statuses[key] = 1
            else:
                fingers_statuses[key] = 0
        fstatus = np.array(list(fingers_statuses.values()))
        #print("LeftHand",results.multi_hand_landmarks[0])
        #print(fstatus)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        #print("face",face)
        supportValues = np.array([count['RIGHT'], count['LEFT'], distance, angle_between_fingers])
        
        # Return the output image, the status of each finger and the count of the fingers up of both hands.
        return output_image,np.concatenate([pose, face, lh, rh,fstatus, supportValues]) 

cap = cv2.VideoCapture("VideosFinal/English/takecare.mp4")
cap.set(3,648)
cap.set(4,488)
cap.set(cv2.CAP_PROP_FPS,60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        print(frame)
        # Check if frame is valid
        if not ret:
            # Set video file position to first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)
        image = segmentor.removeBG(image,(137, 207, 240), threshold= 0.1)
        # Draw landmarks on the original image
        draw_styled_landmarks(image, results) 

        
        image,keypoints = detectFeatures(image, results,display=False)    

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()