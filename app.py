import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import numpy as np
from scipy.spatial.distance import cosine
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from typing import List
import json

app = FastAPI(title="Speaker Classification API", version="1.0.0")

# Global variable for the classifier
classifier = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global classifier
    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_audio(audio_path, target_sr=16000):
    """Load audio file and resample to target sample rate"""
    signal, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        signal = resampler(signal)
    
    # Convert to mono if stereo
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    
    return signal.squeeze()

def extract_embedding(audio_signal):
    """Extract speaker embedding using ECAPA-TDNN"""
    # Add batch dimension
    audio_signal = audio_signal.unsqueeze(0)
    
    # Extract embedding
    with torch.no_grad():
        embedding = classifier.encode_batch(audio_signal)
    
    return embedding.squeeze().cpu().numpy()

def compute_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings"""
    similarity = 1 - cosine(embedding1, embedding2)
    return float(similarity)  # Convert to Python float

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

@app.post("/classify/")
async def classify_speaker(
    reference_audio: UploadFile = File(...),
    comparison_audios: List[UploadFile] = File(...),
    threshold: float = 0.25
):
    """
    Classify if reference audio matches all comparison audios
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Save reference audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as ref_temp:
            ref_temp.write(await reference_audio.read())
            ref_path = ref_temp.name
        
        # Process reference audio
        signal1 = load_audio(ref_path)
        embedding1 = extract_embedding(signal1)
        
        results = []
        cheat_detected = False
        
        # Process each comparison audio
        for comp_audio in comparison_audios:
            try:
                # Save comparison audio to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as comp_temp:
                    comp_temp.write(await comp_audio.read())
                    comp_path = comp_temp.name
                
                # Process comparison audio
                signal2 = load_audio(comp_path)
                embedding2 = extract_embedding(signal2)
                similarity = compute_similarity(embedding1, embedding2)
                
                results.append({
                    "filename": comp_audio.filename,
                    "similarity": similarity,  # Already converted to float
                    "match": bool(similarity > threshold)  # Convert to Python bool
                })
                
                # Clean up temporary file
                os.unlink(comp_path)
                
                if similarity <= threshold:
                    cheat_detected = True
                    
            except Exception as e:
                results.append({
                    "filename": comp_audio.filename,
                    "error": str(e),
                    "match": False
                })
                cheat_detected = True
        
        # Clean up reference temporary file
        os.unlink(ref_path)
        
        response_data = {
            "result": "Cheat" if cheat_detected else "Pass",
            "threshold": float(threshold),
            "comparisons": results,
            "reference_audio": reference_audio.filename
        }
        
        # Convert all numpy types to Python native types
        response_data = convert_to_serializable(response_data)
        
        return JSONResponse(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": bool(classifier)}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Speaker Classification API",
        "endpoints": {
            "POST /classify": "Classify speaker audio",
            "GET /health": "Health check",
            "GET /": "This information"
        }
    }