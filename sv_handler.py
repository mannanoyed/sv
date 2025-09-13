import os
import runpod
import torch
import torchaudio
import requests
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from scipy.spatial.distance import cosine

# -----------------------
# Load model globally
# -----------------------
MODEL_DIR = "/tmp/spkrec_model"
os.makedirs(MODEL_DIR, exist_ok=True)

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=MODEL_DIR
)

# -----------------------
# Helper: download audio
# -----------------------
def download_audio(url, filename):
    local_path = os.path.join("/tmp", filename)
    r = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(r.content)
    return local_path

# -----------------------
# Helper: get embedding
# -----------------------
def get_embedding(audio_path):
    signal, fs = torchaudio.load(audio_path)
    embedding = classifier.encode_batch(signal)
    return embedding.squeeze().detach().cpu().numpy()

# -----------------------
# Core verify function
# -----------------------
def verify_speaker(reference_path, test_path, threshold=0.7):
    ref_emb = get_embedding(reference_path)
    test_emb = get_embedding(test_path)

    sim = 1 - cosine(ref_emb, test_emb)  # similarity score
    return sim >= threshold

# -----------------------
# RunPod handler
# -----------------------
def handler(event):
    try:
        input_data = event["input"]
        ref_url = input_data["audio1"]
        test_urls = input_data["audio_array"]

        # download reference
        ref_path = download_audio(ref_url, "ref.wav")

        # check each test audio
        for i, test_url in enumerate(test_urls):
            test_path = download_audio(test_url, f"test_{i}.wav")
            if not verify_speaker(ref_path, test_path):
                return {"result": "cheat"}

        return {"result": "pass"}

    except Exception as e:
        return {"error": str(e)}

# Start worker loop
runpod.serverless.start({"handler": handler})
