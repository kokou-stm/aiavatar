import requests
import numpy as np
import scipy.io.wavfile as wav

# Générer un fichier audio de silence ou simple sinusoïde pour le test
# (Whisper va juste ne rien entendre ou halluciner, mais ça teste le pipeline)
sample_rate = 44100
duration = 2  # seconds
frequency = 440  # Hz
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
wav.write("test_audio.wav", sample_rate, (audio_data * 32767).astype(np.int16))

url = "http://127.0.0.1:5000/audio-to-landmarks"
files = {'audio': open('test_audio.wav', 'rb')}

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:", response.json())
except Exception as e:
    print(f"An error occurred: {e}")
