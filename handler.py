import runpod
import torch
import torchaudio as ta
import tempfile
import base64
import io
import os
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import the correct module - this might be the issue!
try:
    from chatterbox.tts import ChatterboxTTS
except ImportError:
    # Try alternative import paths
    try:
        from src.tts import ChatterboxTTS
    except ImportError:
        from ChatterboxTTS import ChatterboxTTS

# Global model variable for caching
model = None

def initialize_model():
    """Initialize the Chatterbox TTS model once."""
    global model
    if model is None:
        try:
            print("🔥 Loading Chatterbox TTS model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # FIXED: Use the EXACT initialization that works locally
            model = ChatterboxTTS.from_pretrained(device=device)
            
            print(f"✅ Model loaded successfully on {device}")
            return model
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()
            model = None
    return model

def decode_audio_prompt(audio_prompt):
    """Safely decode and save audio prompt to temporary file"""
    try:
        print("🎤 Processing voice cloning audio...")
        
        # Handle different base64 formats
        if audio_prompt.startswith('data:audio'):
            # Remove data URL prefix if present
            audio_prompt = audio_prompt.split(',')[1]
        
        # Decode base64 audio
        audio_data = base64.b64decode(audio_prompt)
        print(f"📦 Decoded audio data: {len(audio_data)} bytes")
        
        # Save to temporary file with proper extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        print(f"💾 Voice sample saved to: {temp_file_path}")
        return temp_file_path
        
    except Exception as e:
        print(f"⚠️ Audio processing failed: {str(e)}")
        return None

def handler(event):
    """RunPod serverless handler for Chatterbox TTS with voice cloning"""
    try:
        print("🚀 Processing TTS request...")
        
        # Initialize model with error checking
        tts_model = initialize_model()
        if tts_model is None:
            return {"error": "Failed to load TTS model"}
        
        # Extract input parameters
        input_data = event.get('input', {})
        text = input_data.get('text', '')
        emotion_level = input_data.get('emotion_level', 0.5)
        audio_prompt = input_data.get('audio_prompt')
        
        # Validation
        if not text:
            return {"error": "No text provided for synthesis"}
        
        print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"🎭 Emotion level: {emotion_level}")
        print(f"🎤 Voice cloning: {'Yes' if audio_prompt else 'No'}")
        
        # Process voice cloning audio if provided
        audio_prompt_path = None
        if audio_prompt:
            audio_prompt_path = decode_audio_prompt(audio_prompt)
            if not audio_prompt_path:
                print("⚠️ Voice cloning failed, using default voice")
        
        # Generate speech with proper error handling
        try:
            if audio_prompt_path:
                print("🎵 Generating with voice cloning...")
                # FIXED: Use the EXACT method that works locally
                wav = tts_model.generate(
                    text=text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=emotion_level
                )
                
            else:
                print("🎵 Generating with default voice...")
                wav = tts_model.generate(text, exaggeration=emotion_level)
                
        except Exception as generation_error:
            print(f"❌ Generation failed: {str(generation_error)}")
            # Fallback to basic generation
            print("🔄 Falling back to basic generation...")
            wav = tts_model.generate(text)
        
        finally:
            # Clean up temporary file
            if audio_prompt_path and os.path.exists(audio_prompt_path):
                try:
                    os.unlink(audio_prompt_path)
                    print("🗑️ Cleaned up temporary audio file")
                except:
                    pass
        
        # Validate output
        if wav is None:
            return {"error": "Audio generation returned None"}
            
        print(f"🎵 Generated audio shape: {wav.shape}")
        
        # Convert to base64 audio
        try:
            buffer = io.BytesIO()
            
            # Ensure wav has correct dimensions for torchaudio
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            elif wav.dim() == 3:
                wav = wav.squeeze(0)  # Remove batch dimension if present
            
            # Use model's sample rate or default
            sample_rate = getattr(tts_model, 'sr', 24000)
            
            # Save to buffer
            ta.save(buffer, wav.cpu(), sample_rate, format="wav")
            buffer.seek(0)
            audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            print(f"✅ Audio converted to base64: {len(audio_b64)} characters")
            
        except Exception as audio_error:
            print(f"❌ Audio conversion failed: {str(audio_error)}")
            return {"error": f"Audio conversion failed: {str(audio_error)}"}
        
        # Return response in expected format
        response = {
            "output": {
                "audio_base64": audio_b64,
                "duration_seconds": float(len(wav[0]) / sample_rate),
                "sample_rate": int(sample_rate),
                "voice_cloned": bool(audio_prompt),
                "emotion_level": emotion_level
            },
            "text": text,
            "status": "success"
        }
        
        print("🎉 TTS generation completed successfully!")
        return response
        
    except Exception as e:
        print(f"💥 Handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"TTS generation failed: {str(e)}",
            "status": "error"
        }

if __name__ == "__main__":
    print("🚀 Starting Chatterbox TTS RunPod Serverless Worker...")
    print("🎤 Voice cloning enabled!")
    runpod.serverless.start({"handler": handler})
