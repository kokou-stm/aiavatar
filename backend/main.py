import json
import cv2
import numpy as np

# ====================== CONFIG ======================
JSON_FILE = "quick_test.json"
OUTPUT_VIDEO = "landmarks_video.mp4"
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600
FPS = 30                     # on force 30 fps pour que ça soit fluide même si le .pose est à 25 ou 60
LINE_THICKNESS = 4
POINT_RADIUS = 6
# ===================================================

# Chargement + nettoyage des None → NaN
with open(JSON_FILE, "r", encoding="utf-8") as f:
    pose = json.load(f)

def clean_none(d):
    """Remplace récursivement tous les None par NaN"""
    if isinstance(d, list):
        return [clean_none(x) for x in d]
    elif d is None:
        return float('nan')
    return d

data_list = clean_none(pose["data"])
frames_data = np.array(data_list, dtype=np.float32)   # None → NaN + float
frames_data = frames_data[:, 0, :, :]                 # (frames, keypoints, xyz)

num_frames = len(frames_data)

# Calcul du scaling (robuste aux NaN)
min_x = np.nanmin(frames_data[:, :, 0])
max_x = np.nanmax(frames_data[:, :, 0])
min_y = np.nanmin(frames_data[:, :, 1])
max_y = np.nanmax(frames_data[:, :, 1])

scale_x = VIDEO_WIDTH / (max_x - min_x) if max_x > min_x else 1
scale_y = VIDEO_HEIGHT / (max_y - min_y) if max_y > min_y else 1
offset_x = -min_x * scale_x
offset_y = -min_y * scale_y

# Création vidéo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

print(f"Création de la vidéo ({num_frames} frames)...")

for f_idx in range(num_frames):
    frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
    pts = frames_data[f_idx]                              # (keypoints, 3)

    kp_offset = 0
    for comp in pose["header"]["components"]:
        num_pts = len(comp["points"])
        comp_pts = pts[kp_offset : kp_offset + num_pts]
        kp_offset += num_pts

        # Couleurs
        name = comp["name"].lower()
        if "hand" in name:
            color = (0, 0, 255)      # rouge
        elif "face" in name:
            color = (0, 255, 0)      # vert
        else:
            color = (255, 100, 0)    # orange

        # Lignes (limbs)
        for limb in comp.get("limbs", []):
            if len(limb) != 2:
                continue
            i1, i2 = limb
            p1 = comp_pts[i1]
            p2 = comp_pts[i2]

            if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                continue

            x1 = int(p1[0] * scale_x + offset_x)
            y1 = int(p1[1] * scale_y + offset_y)
            x2 = int(p2[0] * scale_x + offset_x)
            y2 = int(p2[1] * scale_y + offset_y)

            cv2.line(frame, (x1, y1), (x2, y2), color, LINE_THICKNESS)

        # Points
        for i, p in enumerate(comp_pts):
            if np.any(np.isnan(p)):
                continue
            x = int(p[0] * scale_x + offset_x)
            y = int(p[1] * scale_y + offset_y)
            cv2.circle(frame, (x, y), POINT_RADIUS, color, -1)

    out.write(frame)

out.release()

print(f" Vidéo terminée → {OUTPUT_VIDEO}")
print(f"   {num_frames} frames • {VIDEO_WIDTH}x{VIDEO_HEIGHT} • {FPS} fps")