import socket
import json
import time
from pose_format import Pose

# Config
UDP_IP = "127.0.0.1"      # IP Unity
UDP_PORT = 12345
POSE_FILE = "quick_test.pose"   # ton fichier .pose
FPS = 25                  # on simule 25 fps

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

with open(POSE_FILE, "rb") as f:
    pose = Pose.read(f.read())

print(f"Envoyé {len(pose.body.data)} frames à {FPS} fps...")

for frame_idx in range(len(pose.body.data)):
    frame = pose.body.data [0].tolist()  # 1 personne, keypoints, xyz
    msg = json.dumps(frame)                       # juste les sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))
    time.sleep(1 / FPS)  # rythme

print("Fin envoyée.")
sock.close()