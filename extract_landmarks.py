import cv2
import mediapipe as mp
import numpy as np
import socket
import json, struct
from dotenv import load_dotenv
import os

load_dotenv()

#User Datagram Protocol
UDP_IP = os.getenv("UDP_IP")
UDP_PORT = int(os.getenv("UDP_PORT"))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

 
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

 
cap = cv2.VideoCapture(0)

 
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'p' to print current landmarks in the terminal.")
print("Press 'q' to quit.")

while cap.isOpened():
     
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

   
    original_frame = frame.copy()

 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

 
    results = holistic.process(rgb_frame)

  
    landmark_frame = np.zeros_like(frame)


 
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
        landmark_frame,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
        landmark_frame,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.
        get_default_pose_landmarks_style())
    
    if results.left_hand_landmarks:
        
        mp_drawing.draw_landmarks(landmark_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(landmark_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    

 
    cv2.imshow('Original Camera Feed', original_frame)

   
    cv2.imshow('Holistic Landmarks', landmark_frame)

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    data = np.concatenate([pose, face, lh, rh]).astype(np.float32)
    print("Landmark Data Shape:", data.shape)
    

    message = struct.pack(f'{len(data)}f', *data)
    sock.sendto(message, (UDP_IP, UDP_PORT))


 
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        print("\nCurrent Landmarks:")

        
        if results.pose_landmarks:
            print("Pose Landmarks:")
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                print(f"  {idx}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}, visibility={lm.visibility:.4f}")
        
        if results.face_landmarks:
            print("Face Landmarks (first 10 for brevity):")
            for idx, lm in enumerate(results.face_landmarks.landmark[:10]):  # Limit to avoid flooding
                print(f"  {idx}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}, visibility={lm.visibility:.4f}")
            if len(results.face_landmarks.landmark) > 10:
                print("  ... (total face landmarks: {})".format(len(results.face_landmarks.landmark)))
        
        if results.left_hand_landmarks:
            print("Left Hand Landmarks:")
            for idx, lm in enumerate(results.left_hand_landmarks.landmark):
                print(f"  {idx}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}, visibility={lm.visibility:.4f}")
        
        if results.right_hand_landmarks:
            print("Right Hand Landmarks:")
            for idx, lm in enumerate(results.right_hand_landmarks.landmark):
                print(f"  {idx}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}, visibility={lm.visibility:.4f}")

 
cap.release()
cv2.destroyAllWindows()