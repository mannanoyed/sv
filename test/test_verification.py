import requests
import json
import time
import os
from pathlib import Path

def test_speaker_verification():
    base_url = "http://localhost:8000"
    
    # Test files directory
    test_dir = "test_audio"
    os.makedirs(test_dir, exist_ok=True)
    
    print("Testing Speaker Verification API...")
    
    # Test 1: Health check
    print("\n1. Testing health endpoint:")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
        return False
    
    # Test 2: Create test audio files
    print("\n2. Creating test audio files...")
    create_test_audio_files(test_dir)
    
    # Test 3: Same speaker (should return "Pass")
    print("\n3. Testing same speaker verification:")
    try:
        files = {
            'reference_audio': open(f'{test_dir}/reference.wav', 'rb'),
            'comparison_audios': [
                open(f'{test_dir}/same_speaker1.wav', 'rb'),
                open(f'{test_dir}/same_speaker2.wav', 'rb')
            ]
        }
        data = {'threshold': 0.25}
        
        response = requests.post(f"{base_url}/classify/", files=files, data=data)
        result = response.json()
        
        print(f"   Status: {response.status_code}")
        print(f"   Result: {result['result']}")
        print(f"   Similarities: {[f'{comp['filename']}: {comp['similarity']:.3f}' for comp in result['comparisons']]}")
        
        # Close files
        for file in files.values():
            if hasattr(file, 'close'):
                file.close()
                if isinstance(file, list):
                    for f in file:
                        f.close()
                        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Different speaker (should return "Cheat")
    print("\n4. Testing different speaker detection:")
    try:
        files = {
            'reference_audio': open(f'{test_dir}/reference.wav', 'rb'),
            'comparison_audios': [
                open(f'{test_dir}/different_speaker.wav', 'rb')
            ]
        }
        data = {'threshold': 0.25}
        
        response = requests.post(f"{base_url}/classify/", files=files, data=data)
        result = response.json()
        
        print(f"   Status: {response.status_code}")
        print(f"   Result: {result['result']}")
        print(f"   Similarities: {[f'{comp['filename']}: {comp['similarity']:.3f}' for comp in result['comparisons']]}")
        
        # Close files
        for file in files.values():
            if hasattr(file, 'close'):
                file.close()
                if isinstance(file, list):
                    for f in file:
                        f.close()
                        
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✓ All tests completed!")

def create_test_audio_files(test_dir):
    """Create simple test audio files with different frequencies"""
    import numpy as np
    import soundfile as sf
    
    sample_rate = 16000
    duration = 3.0
    
    # Reference speaker (440Hz - A4 note)
    t = np.linspace(0, duration, int(sample_rate * duration))
    reference_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(f'{test_dir}/reference.wav', reference_signal, sample_rate)
    
    # Same speaker with slight variation (442Hz)
    same_speaker1 = 0.5 * np.sin(2 * np.pi * 442 * t)
    sf.write(f'{test_dir}/same_speaker1.wav', same_speaker1, sample_rate)
    
    # Same speaker with different phrase (445Hz)
    same_speaker2 = 0.5 * np.sin(2 * np.pi * 445 * t)
    sf.write(f'{test_dir}/same_speaker2.wav', same_speaker2, sample_rate)
    
    # Different speaker (880Hz - A5 note)
    different_speaker = 0.5 * np.sin(2 * np.pi * 880 * t)
    sf.write(f'{test_dir}/different_speaker.wav', different_speaker, sample_rate)

if __name__ == "__main__":
    test_speaker_verification()