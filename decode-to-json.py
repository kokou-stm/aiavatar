import json
import numpy as np
from pose_format import Pose

# ──────────────────────────────────────────────────────────────
# Fonction qui transforme TOUT numpy en JSON (la cause de ton erreur)
# ──────────────────────────────────────────────────────────────
def make_json_friendly(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_friendly(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_friendly(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj

# ──────────────────────────────────────────────────────────────
# Charge ton fichier
# ──────────────────────────────────────────────────────────────
with open("quick_test.pose", "rb") as f:
    pose = Pose.read(f.read())

# ──────────────────────────────────────────────────────────────
# Construction du JSON
# ──────────────────────────────────────────────────────────────
pose_dict = {
    "fps": pose.body.fps,
    "header": {
        "version": pose.header.version,
        "dimensions": {
            "width": pose.header.dimensions.width,
            "height": pose.header.dimensions.height,
            "depth": pose.header.dimensions.depth,
        },
        "components": [
            {
                "name": comp.name,
                "points": comp.points,
                "limbs": [list(limb) for limb in getattr(comp, "limbs", [])],
                "colors": getattr(comp, "colors", None),
                "point_format": getattr(comp, "point_format", None),
            }
            for comp in pose.header.components
        ],
    },
    "data": pose.body.data.tolist(),                    # (frames, people=1, keypoints, xyz)
    "confidence": (
        pose.body.confidence.tolist()
        if hasattr(pose.body, "confidence") and pose.body.confidence is not None
        else None
    ),
}

# Conversion finale
pose_dict = make_json_friendly(pose_dict)

# Sauvegarde
with open("quick_test.json", "w", encoding="utf-8") as f:
    json.dump(pose_dict, f, indent=2, ensure_ascii=False)

print("✅ JSON généré avec succès → quick_test.json")
print(f"   → {len(pose_dict['data'])} frames")
print(f"   → {sum(len(c['points']) for c in pose_dict['header']['components'])} keypoints")