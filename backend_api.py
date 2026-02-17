from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import whisper
import os
import tempfile
import logging
import uvicorn
from pydantic import BaseModel
import json
import subprocess
import numpy as np
import cv2
from langdetect import detect, LangDetectException
from pose_format import Pose
from deep_translator import GoogleTranslator
import socket
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper Voice Recognition API", version="1.0.0")

# Configuration CORS pour permettre les requêtes depuis Unity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle Whisper (vous pouvez changer la taille du modèle)
# Options: tiny, base, small, medium, large
MODEL_SIZE = "base"
logger.info(f"Chargement du modèle Whisper '{MODEL_SIZE}'...")
model = whisper.load_model(MODEL_SIZE)
logger.info("Modèle Whisper chargé avec succès!")


# Modèles Pydantic pour les réponses
class TranscriptionResponse(BaseModel):
    text: str
    language: str


class HealthResponse(BaseModel):
    status: str
    model: str


class TextToVideoRequest(BaseModel):
    text: str
    translate: bool = True
    auto_detect_language: bool = True
    source_language: str = "auto"
    target_language: str = "de"
    spoken_language: str = "de"
    signed_language: str = "sgg"
    lexicon_path: str = "assets/dummy_lexicon"
    fps: int = 30
    width: int = 800
    height: int = 600
    line_thickness: int = 4
    point_radius: int = 6


# UDP Configuration
UDP_IP = os.getenv("UDP_IP", "127.0.0.1")
UDP_PORT = int(os.getenv("UDP_PORT", "5052"))
UDP_REPLAY_PORT = int(os.getenv("UDP_REPLAY_PORT", "5053"))
UDP_TEXT_PORT = int(os.getenv("UDP_TEXT_PORT", "5054"))

# MediaPipe Constants
POSE_LANDMARKS = 33
FACE_LANDMARKS = 468
HAND_LANDMARKS = 21
POSE_FLOATS = POSE_LANDMARKS * 4
FACE_FLOATS = FACE_LANDMARKS * 3
HAND_FLOATS = HAND_LANDMARKS * 3
TOTAL_FLOATS = POSE_FLOATS + FACE_FLOATS + HAND_FLOATS * 2

# Mapping pose indices MediaPipe
POSE_MAP = [11, 12, 13, 14, 15, 16, 23, 24]  # LEFT_SHOULDER -> RIGHT_HIP


# Helper function to make JSON friendly
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


def clean_none(obj):
    if isinstance(obj, list):
        return [clean_none(x) for x in obj]
    if obj is None:
        return float("nan")
    return obj


def generate_landmarks_video(
    pose_dict: dict,
    output_path: str,
    width: int,
    height: int,
    fps: int,
    line_thickness: int,
    point_radius: int,
):
    data_list = clean_none(pose_dict["data"])
    frames_data = np.array(data_list, dtype=np.float32)
    frames_data = frames_data[:, 0, :, :]

    min_x = np.nanmin(frames_data[:, :, 0])
    max_x = np.nanmax(frames_data[:, :, 0])
    min_y = np.nanmin(frames_data[:, :, 1])
    max_y = np.nanmax(frames_data[:, :, 1])

    scale_x = width / (max_x - min_x) if max_x > min_x else 1
    scale_y = height / (max_y - min_y) if max_y > min_y else 1
    offset_x = -min_x * scale_x
    offset_y = -min_y * scale_y

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for f_idx in range(len(frames_data)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        pts = frames_data[f_idx]

        kp_offset = 0
        for comp in pose_dict["header"]["components"]:
            num_pts = len(comp["points"])
            comp_pts = pts[kp_offset : kp_offset + num_pts]
            kp_offset += num_pts

            name = comp["name"].lower()
            if "hand" in name:
                color = (0, 0, 255)
            elif "face" in name:
                color = (0, 255, 0)
            else:
                color = (255, 100, 0)

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

                cv2.line(frame, (x1, y1), (x2, y2), color, line_thickness)

            for p in comp_pts:
                if np.any(np.isnan(p)):
                    continue
                x = int(p[0] * scale_x + offset_x)
                y = int(p[1] * scale_y + offset_y)
                cv2.circle(frame, (x, y), point_radius, color, -1)

        out.write(frame)

    out.release()


def build_pose_dict(pose: Pose) -> dict:
    return {
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
        "data": pose.body.data.tolist(),
        "confidence": (
            pose.body.confidence.tolist()
            if hasattr(pose.body, "confidence") and pose.body.confidence is not None
            else None
        ),
    }


def resolve_lexicon_path(cwd: str, lexicon_path: str) -> str:
    resolved = lexicon_path
    if not resolved or resolved.strip().lower() == "string":
        resolved = "assets/dummy_lexicon"
    if not os.path.isabs(resolved):
        resolved = os.path.join(cwd, resolved)
    return resolved


def maybe_translate_text(payload: TextToVideoRequest) -> str:
    input_text = payload.text
    if payload.translate:
        source_language = payload.source_language
        if payload.auto_detect_language:
            try:
                source_language = detect(input_text)
            except LangDetectException:
                logger.warning("LangDetect a échoué, fallback sur source_language")
        translator = GoogleTranslator(
            source=source_language,
            target=payload.target_language,
        )
        input_text = translator.translate(input_text)
        logger.info(f"Traduction ({source_language}->{payload.target_language}): {input_text}")
    return input_text


def stream_landmarks_to_unity(
    pose_dict: dict,
    udp_ip: str = UDP_IP,
    udp_port: int = UDP_PORT,
):
    """
    Stream landmarks via UDP to Unity
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        data = np.array(pose_dict["data"])  # (frames, 1, 178, 3)
        frames = data.shape[0]
        
        # Header pour mapping
        header = pose_dict["header"]
        width = header["dimensions"]["width"]
        height = header["dimensions"]["height"]
        depth = header["dimensions"]["depth"]
        
        # Face points mapping
        face_points = [int(p) for p in header["components"][1]["points"]]
        
        # Confidence
        has_conf = "confidence" in pose_dict
        if has_conf:
            conf = np.array(pose_dict["confidence"])
        
        logger.info(f"Streaming {frames} frames via UDP to {udp_ip}:{udp_port}")
        
        for frame_idx in range(frames):
            # Extraire keypoints du frame (178, 3)
            kp = data[frame_idx, 0].astype(np.float32)
            
            # Normaliser (0-1 comme MediaPipe)
            kp[:, 0] /= width   # x
            kp[:, 1] /= height  # y
            kp[:, 2] /= depth   # z
            
            # Flat array pour UDP
            flat = np.zeros(TOTAL_FLOATS, dtype=np.float32)
            
            # POSE (33*4 : seulement 8 remplis)
            for j, mp_idx in enumerate(POSE_MAP):
                x, y, z = kp[j]
                vis = conf[frame_idx, 0, j] if has_conf else 1.0
                flat[mp_idx * 4 : mp_idx * 4 + 3] = [x, y, z]
                flat[mp_idx * 4 + 3] = vis
            
            # FACE (468*3 : subset 128 aux bons indices)
            face_start = 8
            for j, mp_idx in enumerate(face_points):
                x, y, z = kp[face_start + j]
                flat[POSE_FLOATS + mp_idx * 3 : POSE_FLOATS + mp_idx * 3 + 3] = [x, y, z]
            
            # LEFT HAND (21*3)
            lh_start = 8 + 128
            flat[POSE_FLOATS + FACE_FLOATS : POSE_FLOATS + FACE_FLOATS + HAND_FLOATS] = kp[lh_start : lh_start + 21].flatten()
            
            # RIGHT HAND (21*3)
            rh_start = lh_start + 21
            flat[POSE_FLOATS + FACE_FLOATS + HAND_FLOATS : ] = kp[rh_start : ].flatten()
            
            # Envoi UDP
            message = flat.tobytes()
            sock.sendto(message, (udp_ip, udp_port))
            
            # FPS control
            import time
            time.sleep(1.0 / pose_dict["fps"])
        
        sock.close()
        logger.info(f"✅ Streaming terminé: {frames} frames envoyés")

        
    except Exception as e:
        logger.error(f"Erreur lors du streaming UDP: {str(e)}")
        raise


def send_text_to_unity(text: str, udp_ip: str = UDP_IP, udp_port: int = UDP_TEXT_PORT):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        payload = text.encode('utf-8')
        sock.sendto(payload, (udp_ip, udp_port))
        sock.close()
        logger.info(f"Texte envoyé via UDP vers {udp_ip}:{udp_port}")
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi texte UDP: {str(e)}")
        raise


@app.post("/transcribe1", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Endpoint pour recevoir l'audio et retourner la transcription
    
    Args:
        audio: Fichier audio au format WAV
        
    Returns:
        TranscriptionResponse: Texte transcrit et langue détectée
    """
    try:
        # Vérifier que le fichier est bien un audio
        if not audio.filename:
            raise HTTPException(status_code=400, detail="Nom de fichier vide")
        
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            content = await audio.read()
            temp_file.write(content)
            logger.info(f"Fichier audio sauvegardé: {temp_path} ({len(content)} bytes)")
        
        # Transcrire avec Whisper
        logger.info("Début de la transcription...")
        result = model.transcribe(temp_path, language='fr')  # Changez 'fr' pour autre langue si nécessaire
        
        transcription = result['text'].strip()
        detected_language = result.get('language', 'unknown')
        logger.info(f"Transcription: {transcription}")
        
        # Supprimer le fichier temporaire
        os.unlink(temp_path)
        
        # Retourner la transcription
        return TranscriptionResponse(
            text=transcription,
            language=detected_language
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe", response_model=TranscriptionResponse)
async def audio_to_landmarks(
    audio: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Endpoint pour recevoir l'audio directement depuis Unity, le transcrire, 
    le traduire et générer les landmarks (JSON).
    Sauvegarde le résultat dans 'generated_landmarks.json'
    """
    try:
        # 1. Sauvegarder temporairement le fichier audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio_path = temp_audio.name
            content = await audio.read()
            temp_audio.write(content)
            logger.info(f"Fichier audio sauvegardé: {temp_audio_path}")

        # 2. Transcrire avec Whisper (France)
        logger.info("Début de la transcription...")
        result = model.transcribe(temp_audio_path, language='fr')
        transcription = result['text'].strip()
        detected_language = result.get('language', 'unknown')
        logger.info(f"Transcription: {transcription}")

        # 3. Traduire (FR -> DE) avec deep-translator
        translator = GoogleTranslator(source='fr', target='de')
        translated_text = translator.translate(transcription)
        logger.info(f"Traduction (DE): {translated_text}")

        # 4. Générer la Pose via text_to_gloss_to_pose
        # On utilise un fichier temporaire pour la pose
        temp_pose_path = temp_audio_path + ".pose"
        
        # Chemin absolu vers le lexique
        cwd = os.getcwd()
        lexicon_path = os.path.join(cwd, "assets", "dummy_lexicon")

        cmd = [
            "text_to_gloss_to_pose",
            "--text", translated_text,
            "--glosser", "simple",
            "--lexicon", lexicon_path,
            "--spoken-language", "de",
            "--signed-language", "sgg",
            "--pose", temp_pose_path,
        ]

        logger.info(f"Lancement de la génération de pose: {' '.join(cmd)}")
        process_result = subprocess.run(cmd, capture_output=True, text=True)

        if process_result.returncode != 0:
            logger.error(f"Erreur text_to_gloss_to_pose: {process_result.stderr}")
            raise HTTPException(status_code=500, detail=f"Erreur génération pose: {process_result.stderr}")

        logger.info("Pose générée avec succès.")

        # 5. Convertir .pose en JSON
        with open(temp_pose_path, "rb") as f:
            pose = Pose.read(f.read())

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
            "data": pose.body.data.tolist(),
            "confidence": (
                pose.body.confidence.tolist()
                if hasattr(pose.body, "confidence") and pose.body.confidence is not None
                else None
            ),
        }

        final_json = make_json_friendly(pose_dict)

        # 6. Sauvegarder le JSON en local
        output_filename = "generated_landmarks.json"
        output_path = os.path.join(cwd, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON sauvegardé dans : {output_path}")

        # 7. Stream landmarks to Unity via UDP
        logger.info("Début du streaming UDP vers Unity...")
        stream_landmarks_to_unity(final_json, UDP_IP, UDP_REPLAY_PORT)

        # 8. Send text to Unity via UDP (optional)
        if background_tasks is not None:
            background_tasks.add_task(
                send_text_to_unity,
                transcription,
                UDP_IP,
                UDP_TEXT_PORT,
            )

        # Nettoyage
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        if os.path.exists(temp_pose_path):
            os.unlink(temp_pose_path)

        # Retourner la même réponse que /transcribe
        return TranscriptionResponse(
            text=transcription,
            language=detected_language
        )

    except Exception as e:
        logger.error(f"Erreur globale: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text-to-landmarks-video")
async def text_to_landmarks_video(
    payload: TextToVideoRequest,
    background_tasks: BackgroundTasks,
):
    try:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Texte vide")

        cwd = os.getcwd()
        lexicon_path = resolve_lexicon_path(cwd, payload.lexicon_path)
        if not os.path.isdir(lexicon_path):
            raise HTTPException(status_code=400, detail=f"Lexicon introuvable: {lexicon_path}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pose") as temp_pose:
            temp_pose_path = temp_pose.name

        input_text = maybe_translate_text(payload)

        background_tasks.add_task(
            send_text_to_unity,
            payload.text,
            UDP_IP,
            UDP_TEXT_PORT,
        )

        cmd = [
            "text_to_gloss_to_pose",
            "--text", input_text,
            "--glosser", "simple",
            "--lexicon", lexicon_path,
            "--spoken-language", payload.spoken_language,
            "--signed-language", payload.signed_language,
            "--pose", temp_pose_path,
        ]

        logger.info(f"Lancement de la génération de pose: {' '.join(cmd)}")
        process_result = subprocess.run(cmd, capture_output=True, text=True)
        if process_result.returncode != 0:
            logger.error(f"Erreur text_to_gloss_to_pose: {process_result.stderr}")
            raise HTTPException(status_code=500, detail=f"Erreur génération pose: {process_result.stderr}")

        with open(temp_pose_path, "rb") as f:
            pose = Pose.read(f.read())

        pose_dict = build_pose_dict(pose)

        output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_video_path = output_video.name
        output_video.close()

        generate_landmarks_video(
            pose_dict=pose_dict,
            output_path=output_video_path,
            width=payload.width,
            height=payload.height,
            fps=payload.fps,
            line_thickness=payload.line_thickness,
            point_radius=payload.point_radius,
        )

        background_tasks.add_task(os.unlink, temp_pose_path)
        background_tasks.add_task(os.unlink, output_video_path)

        return FileResponse(
            output_video_path,
            media_type="video/mp4",
            filename="landmarks_video.mp4",
        )

    except Exception as e:
        logger.error(f"Erreur text_to_landmarks_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text-to-landmarks-video-de")
async def text_to_landmarks_video_de(
    payload: TextToVideoRequest,
    background_tasks: BackgroundTasks,
):
    try:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Texte vide")

        cwd = os.getcwd()
        lexicon_path = resolve_lexicon_path(cwd, payload.lexicon_path)
        if not os.path.isdir(lexicon_path):
            raise HTTPException(status_code=400, detail=f"Lexicon introuvable: {lexicon_path}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pose") as temp_pose:
            temp_pose_path = temp_pose.name

        # Pas de traduction : texte allemand brut
        input_text = payload.text

        cmd = [
            "text_to_gloss_to_pose",
            "--text", input_text,
            "--glosser", "simple",
            "--lexicon", lexicon_path,
            "--spoken-language", payload.spoken_language,
            "--signed-language", payload.signed_language,
            "--pose", temp_pose_path,
        ]

        logger.info(f"Lancement de la génération de pose: {' '.join(cmd)}")
        process_result = subprocess.run(cmd, capture_output=True, text=True)
        if process_result.returncode != 0:
            logger.error(f"Erreur text_to_gloss_to_pose: {process_result.stderr}")
            raise HTTPException(status_code=500, detail=f"Erreur génération pose: {process_result.stderr}")

        with open(temp_pose_path, "rb") as f:
            pose = Pose.read(f.read())

        pose_dict = build_pose_dict(pose)

        output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_video_path = output_video.name
        output_video.close()

        generate_landmarks_video(
            pose_dict=pose_dict,
            output_path=output_video_path,
            width=payload.width,
            height=payload.height,
            fps=payload.fps,
            line_thickness=payload.line_thickness,
            point_radius=payload.point_radius,
        )

        background_tasks.add_task(os.unlink, temp_pose_path)
        background_tasks.add_task(os.unlink, output_video_path)

        return FileResponse(
            output_video_path,
            media_type="video/mp4",
            filename="landmarks_video.mp4",
        )

    except Exception as e:
        logger.error(f"Erreur text_to_landmarks_video_de: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text-to-landmarks-json")
async def text_to_landmarks_json(payload: TextToVideoRequest):
    try:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Texte vide")

        cwd = os.getcwd()
        lexicon_path = resolve_lexicon_path(cwd, payload.lexicon_path)
        if not os.path.isdir(lexicon_path):
            raise HTTPException(status_code=400, detail=f"Lexicon introuvable: {lexicon_path}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pose") as temp_pose:
            temp_pose_path = temp_pose.name

        input_text = maybe_translate_text(payload)

        cmd = [
            "text_to_gloss_to_pose",
            "--text", input_text,
            "--glosser", "simple",
            "--lexicon", lexicon_path,
            "--spoken-language", payload.spoken_language,
            "--signed-language", payload.signed_language,
            "--pose", temp_pose_path,
        ]

        logger.info(f"Lancement de la génération de pose: {' '.join(cmd)}")
        process_result = subprocess.run(cmd, capture_output=True, text=True)
        if process_result.returncode != 0:
            logger.error(f"Erreur text_to_gloss_to_pose: {process_result.stderr}")
            raise HTTPException(status_code=500, detail=f"Erreur génération pose: {process_result.stderr}")

        with open(temp_pose_path, "rb") as f:
            pose = Pose.read(f.read())

        pose_dict = build_pose_dict(pose)
        final_json = make_json_friendly(pose_dict)

        if os.path.exists(temp_pose_path):
            os.unlink(temp_pose_path)

        return JSONResponse(content=final_json)

    except Exception as e:
        logger.error(f"Erreur text_to_landmarks_json: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text-to-landmarks-udp")
async def text_to_landmarks_udp(
    payload: TextToVideoRequest,
    background_tasks: BackgroundTasks,
):
    try:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Texte vide")

        cwd = os.getcwd()
        lexicon_path = resolve_lexicon_path(cwd, payload.lexicon_path)
        if not os.path.isdir(lexicon_path):
            raise HTTPException(status_code=400, detail=f"Lexicon introuvable: {lexicon_path}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pose") as temp_pose:
            temp_pose_path = temp_pose.name

        input_text = maybe_translate_text(payload)

        cmd = [
            "text_to_gloss_to_pose",
            "--text", input_text,
            "--glosser", "simple",
            "--lexicon", lexicon_path,
            "--spoken-language", payload.spoken_language,
            "--signed-language", payload.signed_language,
            "--pose", temp_pose_path,
        ]

        logger.info(f"Lancement de la génération de pose: {' '.join(cmd)}")
        process_result = subprocess.run(cmd, capture_output=True, text=True)
        if process_result.returncode != 0:
            logger.error(f"Erreur text_to_gloss_to_pose: {process_result.stderr}")
            raise HTTPException(status_code=500, detail=f"Erreur génération pose: {process_result.stderr}")

        with open(temp_pose_path, "rb") as f:
            pose = Pose.read(f.read())

        pose_dict = build_pose_dict(pose)
        final_json = make_json_friendly(pose_dict)

        background_tasks.add_task(
            stream_landmarks_to_unity,
            final_json,
            UDP_IP,
            UDP_REPLAY_PORT,
        )

        if os.path.exists(temp_pose_path):
            os.unlink(temp_pose_path)

        return {"status": "ok", "frames": len(final_json["data"]), "udp": f"{UDP_IP}:{UDP_REPLAY_PORT}"}

    except Exception as e:
        logger.error(f"Erreur text_to_landmarks_udp: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Endpoint pour vérifier que le serveur fonctionne
    
    Returns:
        HealthResponse: Status du serveur et modèle utilisé
    """
    return HealthResponse(
        status="ok",
        model=MODEL_SIZE
    )


@app.get("/")
async def root():
    """
    Endpoint racine avec informations sur l'API
    """
    return {
        "message": "Whisper Voice Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "POST /transcribe": "Transcrit un fichier audio",
            "POST /audio-to-landmarks": "Transcrit, traduit et génère les landmarks JSON",
            "POST /text-to-landmarks-video": "Prend du texte et génère une vidéo MP4 des landmarks",
            "POST /text-to-landmarks-video-de": "Prend du texte allemand brut et génère une vidéo MP4 des landmarks",
            "POST /text-to-landmarks-json": "Prend du texte et génère les landmarks JSON",
            "POST /text-to-landmarks-udp": "Prend du texte et streame les landmarks par UDP vers Unity",
            "GET /health": "Vérifie l'état du serveur",
            "GET /docs": "Documentation interactive Swagger"
        }
    }


if __name__ == '__main__':
    # Lancer le serveur avec Uvicorn
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=5000,
        reload=True,  # Auto-reload en développement
        log_level="info"
    )
