import subprocess
import sys
from googletrans import Translator

text = "Bonjour, comment ça va ?"
translator = Translator()
translated = translator.translate(text, src='fr', dest='de')
print(f"Texte original : {text}")
print(f"Texte traduit : {translated.text}")

# La commande exacte
cmd = [
    "text_to_gloss_to_pose",
    "--text", translated.text,
    "--glosser", "simple",
    "--lexicon", "assets/dummy_lexicon",
    "--spoken-language", "de",
    "--signed-language", "sgg",
    "--pose", "quick_test.pose"
]

print("Lancement de la génération de pose...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("✅ Pose créée : quick_test.pose")
    print("Logs :\n", result.stdout)
else:
    print("❌ Erreur !")
    print("Sortie :", result.stderr)
    sys.exit(1)