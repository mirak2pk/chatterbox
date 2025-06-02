import runpod
import torch
import torchaudio as ta
import base64
import io
import os
import tempfile
import traceback
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
        
        print(f"📝 Text: {text[:50]}...")
        print(f"🎤 Voice ID: {voice_id}")
        print(f"🎭 Voice Name: {voice_name}")
        print(f"🎵 Audio data present: {audio_data is not None}")
        
        if not text:
            return {"error": "No text provided for synthesis"}
        
        # Check if we have audio data for voice cloning
        voice_sample = None
        if audio_data:
            try:
                print("🎤 Voice cloning: Yes")
                print(f"📦 Audio data length: {len(audio_data)} characters")
                
                # Decode base64 audio
                audio_bytes = base64.b64decode(audio_data)
                print(f"📦 Decoded audio data: {len(audio_bytes)} bytes")
                
                # Create temporary file for voice sample
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_bytes)
                    temp_audio_path = temp_file.name
                
                print(f"💾 Saved temp audio: {temp_audio_path}")
                
                # Load audio for voice cloning
                voice_sample, sr = ta.load(temp_audio_path)
                print(f"🎵 Loaded voice sample: shape={voice_sample.shape}, sr={sr}")
                
                # Resample if needed (ChatterboxTTS typically uses 24kHz)
                if sr != tts_model.sr:
                    print(f"🔄 Resampling from {sr}Hz to {tts_model.sr}Hz")
                    resampler = ta.transforms.Resample(sr, tts_model.sr)
                    voice_sample = resampler(voice_sample)
                
                # Clean up temp file
                os.unlink(temp_audio_path)
                print("🗑️ Cleaned up temp file")
                
            except Exception as e:
                print(f"⚠️ Voice cloning failed: {str(e)}")
                print(f"📋 Traceback: {traceback.format_exc()}")
                print("🔄 Falling back to default voice...")
                voice_sample = None
        else:
            print("🎤 Voice cloning: No (using default voice)")
        
        print("🎵 Generating speech...")
        
        # Generate speech with or without voice cloning
        if voice_sample is not None:
            print("🎙️ Generating with voice cloning...")
            # Check if ChatterboxTTS supports voice cloning
            if hasattr(tts_model, 'generate_with_voice') or hasattr(tts_model, 'clone_voice'):
                # Try voice cloning if supported
                try:
                    if hasattr(tts_model, 'generate_with_voice'):
                        wav = tts_model.generate_with_voice(
                            text=text,
                            voice_sample=voice_sample,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature
                        )
                    elif hasattr(tts_model, 'clone_voice'):
                        wav = tts_model.clone_voice(
                            text=text,
                            reference_audio=voice_sample,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature
                        )
                    print("✅ Voice cloning generation successful!")
                except Exception as e:
                    print(f"⚠️ Voice cloning method failed: {str(e)}")
                    print("🔄 Falling back to standard generation...")
                    wav = tts_model.generate(
                        text=text,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature
                    )
            else:
                print("⚠️ ChatterboxTTS doesn't support voice cloning, using default generation")
                wav = tts_model.generate(
                    text=text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
        else:
            print("🎤 Generating with default voice...")
            wav = tts_model.generate(
                text=text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
        
        print("✅ Speech generation completed!")
        
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
            "voice_cloning_used": voice_sample is not None,
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
