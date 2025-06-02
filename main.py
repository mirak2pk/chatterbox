import runpod
import torch
import torchaudio as ta
import base64
import io
import os
import tempfile
import traceback
import json
from chatterbox.tts import ChatterboxTTS

# Global model variable for caching
model = None

def initialize_model():
    """Initialize the Chatterbox TTS model once."""
    global model
    if model is None:
        try:
            print("🚀 Loading Chatterbox TTS model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🖥️ Using device: {device}")
            model = ChatterboxTTS.from_pretrained(device=device)
            print(f"✅ Model loaded successfully on {device}")
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            model = None
    return model

def handler(event):
    """RunPod serverless handler for Chatterbox TTS with Voice Cloning"""
    try:
        print("🚀 Processing TTS request...")
        
        # Initialize model with error checking
        tts_model = initialize_model()
        if tts_model is None:
            return {"error": "Failed to load TTS model"}
        
        input_data = event.get('input', {})
        text = input_data.get('text', '')
        exaggeration = input_data.get('exaggeration', 0.5)
        cfg_weight = input_data.get('cfg_weight', 0.5)
        temperature = input_data.get('temperature', 0.7)
        
        # Voice cloning parameters
        voice_id = input_data.get('voiceId', 'default')
        voice_name = input_data.get('voiceName', 'Default')
        audio_data = input_data.get('audioData')  # Base64 encoded audio
        
        # Debug logging
        print(f"🔍 DEBUG - Full input_data keys: {list(input_data.keys())}")
        print(f"🔍 DEBUG - audioData type: {type(audio_data)}")
        if audio_data:
            print(f"🔍 DEBUG - audioData length: {len(audio_data)}")
            print(f"🔍 DEBUG - audioData preview (first 50): {audio_data[:50]}")
            print(f"🔍 DEBUG - audioData preview (last 50): {audio_data[-50:]}")
        
        print(f"📝 Text: {text[:50]}...")
        print(f"🎤 Voice ID: {voice_id}")
        print(f"🎭 Voice Name: {voice_name}")
        print(f"🎵 Audio data present: {audio_data is not None}")
        
        if not text:
            return {"error": "No text provided for synthesis"}
        
        # Check if we have audio data for voice cloning
        audio_prompt_path = None
        if audio_data:
            try:
                print("🎤 Voice cloning: Yes")
                print(f"📦 Audio data length: {len(audio_data)} characters")
                
                # Additional base64 validation
                if len(audio_data) < 100:
                    print(f"⚠️ WARNING: Audio data is suspiciously short: {len(audio_data)} chars")
                    print(f"⚠️ Raw audio data: '{audio_data}'")
                    raise Exception(f"Audio data too short: {len(audio_data)} characters")
                
                # Add padding if needed for base64
                missing_padding = len(audio_data) % 4
                if missing_padding:
                    audio_data += '=' * (4 - missing_padding)
                    print(f"🔧 Added {4 - missing_padding} padding characters")
                
                # Decode base64 audio
                audio_bytes = base64.b64decode(audio_data)
                print(f"📦 Decoded audio data: {len(audio_bytes)} bytes")
                
                # Create temporary file for voice sample
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_bytes)
                    audio_prompt_path = temp_file.name
                
                print(f"💾 Saved temp audio: {audio_prompt_path}")
                
                # Load and verify audio
                voice_sample, sr = ta.load(audio_prompt_path)
                print(f"🎵 Loaded voice sample: shape={voice_sample.shape}, sr={sr}")
                
            except Exception as e:
                print(f"⚠️ Voice cloning setup failed: {str(e)}")
                print(f"📋 Traceback: {traceback.format_exc()}")
                print("🔄 Falling back to default voice...")
                if audio_prompt_path and os.path.exists(audio_prompt_path):
                    os.unlink(audio_prompt_path)
                audio_prompt_path = None
        else:
            print("🎤 Voice cloning: No (using default voice)")
        
        print("🎵 Generating speech...")
        
        # Generate speech with ChatterboxTTS
        try:
            if audio_prompt_path:
                print("🎙️ Generating with voice cloning...")
                print(f"🎤 Using audio_prompt_path: {audio_prompt_path}")
                
                # ✅ CORRECT: Use audio_prompt_path parameter
                wav = tts_model.generate(
                    text=text,
                    audio_prompt_path=audio_prompt_path,  # ← This is the correct way!
                    exaggeration=exaggeration,
                    # cfg_weight=cfg_weight,  # Remove if not supported
                    # temperature=temperature  # Remove if not supported
                )
                print("✅ Voice cloning generation successful!")
                
                # Clean up temp file
                os.unlink(audio_prompt_path)
                print("🗑️ Cleaned up temp file")
                
            else:
                print("🎤 Generating with default voice...")
                wav = tts_model.generate(
                    text=text,
                    exaggeration=exaggeration,
                    # cfg_weight=cfg_weight,  # Remove if not supported
                    # temperature=temperature  # Remove if not supported
                )
            
            print("✅ Speech generation completed!")
            
        except Exception as e:
            print(f"❌ Generation failed: {str(e)}")
            print(f"📋 Traceback: {traceback.format_exc()}")
            
            # Clean up temp file if it exists
            if audio_prompt_path and os.path.exists(audio_prompt_path):
                os.unlink(audio_prompt_path)
            
            return {
                "error": f"Speech generation failed: {str(e)}",
                "traceback": traceback.format_exc(),
                "status": "failed"
            }
        
        # Convert to base64 audio
        print("📤 Converting to base64...")
        buffer = io.BytesIO()
        # Ensure wav has correct dimensions
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        ta.save(buffer, wav, tts_model.sr, format="wav")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Calculate duration
        duration = wav.shape[-1] / tts_model.sr
        
        print(f"🎉 Success! Duration: {duration:.2f}s, Audio size: {len(audio_b64)} chars")
        
        return {
            "audio_base64": audio_b64,
            "sample_rate": tts_model.sr,
            "duration_seconds": round(duration, 2),
            "text": text,
            "voice_cloning_used": audio_prompt_path is not None,
            "voice_id": voice_id,
            "voice_name": voice_name,
            "status": "success"
        }
        
    except Exception as e:
        error_msg = f"TTS generation failed: {str(e)}"
        print(f"❌ Error: {error_msg}")
        print(f"📋 Full traceback: {traceback.format_exc()}")
        return {
            "error": error_msg,
            "traceback": traceback.format_exc(),
            "status": "failed"
        }

if __name__ == "__main__":
    print("🚀 Starting Chatterbox TTS RunPod Serverless Worker with Voice Cloning...")
    runpod.serverless.start({"handler": handler})
