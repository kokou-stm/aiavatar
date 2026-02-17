# Traduction Spoken-to-Signed (Texte, Audio, JSON, Vidéo, Unity)

Ce dépôt étend le pipeline **gloss-based spoken-to-signed** avec une API pratique et une intégration Unity :

- **Texte → Gloss → Pose → JSON**
- **Texte → Gloss → Pose → MP4 (landmarks)**
- **Audio → Transcription → Traduction → Pose → UDP vers Unity**
- **Overlay texte via UDP dans Unity**

![Visualization of our pipeline](assets/pipeline.jpg)

---

## Contenu

- **Backend FastAPI** (`backend_api.py`) avec des endpoints pour :
  - transcription audio
  - génération de landmarks JSON
  - rendu MP4 des landmarks
  - streaming UDP vers Unity
- **Scripts Unity** pour :
  - réception UDP des landmarks + affichage du texte
  - bouton Record (rouge/vert), capture micro et upload audio
- **Lexique de test** (`assets/dummy_lexicon`) pour des essais rapides

## Grandes étapes

- Human Input to landmarks
- Display landmarks with Unity
- Text-to-pose
- Text-to-JSON
- Audio-from Unity to landmarks
- Audio-from Unity to landmarks display on Unity

---

## Démarrage rapide (Backend)

```bash
# créer/activer venv
python -m venv venv
source venv/bin/activate

# installer les dépendances
pip install -r requirements.txt

# lancer l'API
uvicorn backend_api:app --host 127.0.0.1 --port 5000
```

L'API sera disponible à :
- Swagger UI : `http://127.0.0.1:5000/docs`

---

## Endpoints

### `POST /transcribe`
Upload d'un audio (`wav`) et retour de la transcription. Génère aussi les landmarks et les envoie vers Unity en UDP.

- Envoie les landmarks vers `UDP_REPLAY_PORT` (par défaut **5053**)
- Envoie le texte vers `UDP_TEXT_PORT` (par défaut **5054**)

### `POST /text-to-landmarks-json`
Envoyer du texte, recevoir un JSON de landmarks.

### `POST /text-to-landmarks-video`
Envoyer du texte, recevoir une vidéo MP4 des landmarks.

### `POST /text-to-landmarks-video-de`
Envoyer du texte allemand brut (sans traduction), recevoir une vidéo MP4 des landmarks.

### `POST /text-to-landmarks-udp`
Envoyer du texte, streamer les landmarks via UDP vers Unity (et envoyer le texte).

---

## Traduction + Détection de langue

Pour les endpoints texte, la langue est auto‑détectée et traduite en allemand avant génération des poses :

- `langdetect` détecte la langue d'entrée
- `deep-translator` traduit vers **DE**

Vous pouvez désactiver ou forcer via le payload :

```json
{
  "translate": true,
  "auto_detect_language": true,
  "source_language": "auto",
  "target_language": "de"
}
```

---

## Intégration Unity

### UDP Receiver (Landmarks + Texte)
Script : `UdpReceiver.cs`

- Reçoit les landmarks sur le port **5053**
- Reçoit le texte sur le port **5054**
- Nécessite un champ `TextMeshPro` pour afficher le texte

### Record & Send (Micro + Bouton)
Script : `RecordAndSendAudio.cs`

- Bouton rouge = enregistrement
- Bouton vert = idle
- Envoie l'audio vers `http://127.0.0.1:5000/transcribe`
- Met à jour le texte transcrit

---

## Variables d’environnement

```bash
UDP_IP=127.0.0.1
UDP_PORT=5052
UDP_REPLAY_PORT=5053
UDP_TEXT_PORT=5054
```

---

## Notes

- Le lexique de test est minimal et ne couvre que quelques mots en allemand.
- En cas d'erreurs de lookup, utilisez ce lexique ou un plus grand.
- Le format UDP suit le layout MediaPipe (pose/face/mains).

---

## Projet d’origine (Upstream)

Ce dépôt repose sur le pipeline **ZurichNLP spoken-to-signed** :
- Paper : https://arxiv.org/abs/2305.17714
- Démo : https://sign.mt

---

## Licence

Voir `LICENSE`.
