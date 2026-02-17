import whisper
import sounddevice as sd
import numpy as np

model = whisper.load_model("base")

duration = 5  # record 5 seconds
fs = 16000

print("Recording...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()

audio = np.squeeze(audio)

result = model.transcribe(audio)
print(result["text"])
