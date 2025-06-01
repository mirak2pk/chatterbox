import runpod
import torch
import torchaudio as ta
import tempfile
import base64
import io
import os
from chatterbox.tts import ChatterboxTTS

# Global model variable for caching
model = None

def initialize_model():
    """Initialize the Chatterbox TTS model once."""
    global model
    if model is None:
        try:
            print("Loading Chatterbox TTS model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            model = ChatterboxTTS.from_pretrained(device=device)
            print(f"Model loaded successfully on {device}")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            model = None
    return model

def handler(event):
    """RunPod serverless handler for Chatterbox TTS"""
    try:
        # Initialize model with error checking
        tts_model = initialize_model()
        if tts_model is None:
            return {"error": "Failed to load TTS model"}
        
        input_data = event.get('input', {})
        text = input_data.get('text', '')
        emotion_level = input_data.get('emotion_level', 0.5)  # NEW: Get emotion
        audio_prompt = input_data.get('audio_prompt')  # NEW: Get voice cloning audio
        
        if not text:
            return {"error": "No text provided for synthesis"}
        
        print(f"Generating TTS for: {text[:50]}...")
        print(f"Emotion level: {emotion_level}")
        
        # NEW: Handle voice cloning if provided
        audio_prompt_path = None
        if audio_prompt:
            print("🎤 Voice cloning requested")
            try:
                # Decode base64 audio
                audio_data = base64.b64decode(audio_prompt)
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(audio_data)
                    audio_prompt_path = temp_file.name
                print(f"✅ Voice sample saved to: {audio_prompt_path}")
            except Exception as e:
                print(f"⚠️ Voice cloning failed, using default: {e}")
                audio_prompt_path = None
        
        # UPDATED: Generate speech with optional voice cloning and emotion
        if audio_prompt_path:
            print("🎵 Generating with voice cloning...")
            wav = tts_model.generate(
                text, 
                audio_prompt_path=audio_prompt_path,
                exaggeration=emotion_level
            )
            # Clean up temp file
            os.unlink(audio_prompt_path)
        else:
            print("🎵 Generating with default voice...")
            wav = tts_model.generate(text, exaggeration=emotion_level)
        
        # Convert to base64 audio (keeping your existing format)
        buffer = io.BytesIO()
        # Ensure wav has correct dimensions
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        ta.save(buffer, wav, tts_model.sr, format="wav")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # UPDATED: Return format matching your frontend expectations
        return {
            "output": {  # Changed from top-level to match frontend
                "audio_base64": audio_b64,  # Changed key name
                "duration_seconds": len(wav[0]) / tts_model.sr,
                "sample_rate": tts_model.sr
            },
            "text": text,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"TTS generation failed: {str(e)}"}

if __name__ == "__main__":
    print("Starting Chatterbox TTS RunPod Serverless Worker...")
    runpod.serverless.start({"handler": handler})
