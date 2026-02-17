import json
import numpy as np
import socket
import time
from dotenv import load_dotenv
import os

load_dotenv()

# UDP Settings (mêmes que extract_landmarks.py)
UDP_IP = os.getenv("UDP_IP", "127.0.0.1")
UDP_PORT = int(os.getenv("UDP_PORT", "5053"))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Constants MediaPipe (comme dans Unity)
POSE_LANDMARKS = 33
FACE_LANDMARKS = 468
HAND_LANDMARKS = 21
POSE_FLOATS = POSE_LANDMARKS * 4
FACE_FLOATS = FACE_LANDMARKS * 3
HAND_FLOATS = HAND_LANDMARKS * 3
TOTAL_FLOATS = POSE_FLOATS + FACE_FLOATS + HAND_FLOATS * 2

# Charge le JSON
with open("quick_test.json", "r", encoding="utf-8") as f:
    pose_dict = json.load(f)

data = np.array(pose_dict["data"])  # (frames, 1, 178, 3) → on aplatit
frames = data.shape[0]
print(f"✅ JSON chargé : {frames} frames @ {pose_dict['fps']} FPS")

# Header pour mapping
header = pose_dict["header"]
width = header["dimensions"]["width"]
height = header["dimensions"]["height"]
depth = header["dimensions"]["depth"]

# Components pour mapping
pose_points = header["components"][0]["points"]  # 8 noms (on mappe manuellement)
face_points = [int(p) for p in header["components"][1]["points"]]  # 128 indices MediaPipe
# Mains : direct 0-20

# Confidence pour vis (optionnel, on l'utilise pour pose)
has_conf = "confidence" in pose_dict
if has_conf:
    conf = np.array(pose_dict["confidence"])  # (frames, 1, 178)
    print("   → Confidence détectée (utilisée pour visibility pose)")

# Mapping pose indices MediaPipe (standard)
POSE_MAP = [11, 12, 13, 14, 15, 16, 23, 24]  # LEFT_SHOULDER -> RIGHT_HIP

print("🚀 Replay UDP démarré (Ctrl+C pour stopper)")

frame_idx = 0
try:
    while True:
        # Extraire keypoints du frame (178, 3)
        kp = data[frame_idx, 0].astype(np.float32)  # (178, 3)
        
        # Normaliser (0-1 comme MediaPipe)
        kp[:, 0] /= width   # x
        kp[:, 1] /= height  # y
        kp[:, 2] /= depth   # z (relatif)
        
        # Flat array pour UDP
        flat = np.zeros(TOTAL_FLOATS, dtype=np.float32)
        
        # ===== POSE (33*4 : seulement 8 remplis) =====
        for j, mp_idx in enumerate(POSE_MAP):
            x, y, z = kp[j]
            vis = conf[frame_idx, 0, j] if has_conf else 1.0
            flat[mp_idx * 4 : mp_idx * 4 + 3] = [x, y, z]
            flat[mp_idx * 4 + 3] = vis
        
        # ===== FACE (468*3 : subset 128 aux bons indices) =====
        face_start = 8  # après les 8 pose
        for j, mp_idx in enumerate(face_points):
            x, y, z = kp[face_start + j]
            flat[POSE_FLOATS + mp_idx * 3 : POSE_FLOATS + mp_idx * 3 + 3] = [x, y, z]
        
        # ===== LEFT HAND (21*3) =====
        lh_start = 8 + 128
        flat[POSE_FLOATS + FACE_FLOATS : POSE_FLOATS + FACE_FLOATS + HAND_FLOATS] = kp[lh_start : lh_start + 21].flatten()
        
        # ===== RIGHT HAND (21*3) =====
        rh_start = lh_start + 21
        flat[POSE_FLOATS + FACE_FLOATS + HAND_FLOATS : ] = kp[rh_start : ].flatten()
        
        # Envoi UDP
        message = flat.tobytes()
        sock.sendto(message, (UDP_IP, UDP_PORT))
        
        # Debug léger (toutes les 25 frames)
        if frame_idx % 25 == 0:
            print(f"📤 Frame {frame_idx}/{frames} envoyé")
        
        # FPS control
        time.sleep(1.0 / pose_dict["fps"])
        
        frame_idx = (frame_idx + 1) % frames  # Loop infini
        
except KeyboardInterrupt:
    print("\n🛑 Replay arrêté")
finally:
    sock.close()