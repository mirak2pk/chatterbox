import runpod
import torch
import torchaudio as ta
import base64
import io
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
        exaggeration = input_data.get('exaggeration', 0.5)
        cfg_weight = input_data.get('cfg_weight', 0.5)
        temperature = input_data.get('temperature', 0.7)
        
        if not text:
            return {"error": "No text provided for synthesis"}
        
        print(f"Generating TTS for: {text[:50]}...")
        
        # Generate speech with parameters
        wav = tts_model.generate(
            text=text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature
        )
        
        # Convert to base64 audio
        buffer = io.BytesIO()
        # Ensure wav has correct dimensions
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        ta.save(buffer, wav, tts_model.sr, format="wav")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Calculate duration
        duration = wav.shape[-1] / tts_model.sr
        
        return {
            "audio_base64": audio_b64,
            "sample_rate": tts_model.sr,
            "duration_seconds": round(duration, 2),
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
