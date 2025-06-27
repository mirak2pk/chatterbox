import runpod
import torch
import torchaudio as ta
import base64
import io
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

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
            model = ChatterboxTTS.from_pretrained("./src/chatterbox/models", device=device, local_files_only=True)
            print(f"✅ Model loaded successfully on {device}")
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            model = None
    return model

def handler(event):
    """
    RunPod serverless handler for Chatterbox TTS
    BACKWARD COMPATIBLE - Works exactly like before + optional .pt file support
    """
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
        
        # EXISTING: Original voice cloning parameters (unchanged)
        voice_id = input_data.get('voiceId', input_data.get('voice_id', 'default'))
        voice_name = input_data.get('voiceName', 'Default')
        audio_data = input_data.get('audioData')  # Base64 encoded audio
        
        # NEW: Optional .pt file support (only if present)
        pt_file_data = input_data.get('pt_file_data')  # Optional: Array of bytes
        
        # Debug logging
        print(f"🔍 DEBUG - Full input_data keys: {list(input_data.keys())}")
        print(f"🔍 DEBUG - audioData type: {type(audio_data)}")
        print(f"🔍 DEBUG - pt_file_data present: {pt_file_data is not None}")
        
        if audio_data:
            print(f"🔍 DEBUG - audioData length: {len(audio_data)}")
        if pt_file_data:
            print(f"🔍 DEBUG - pt_file_data length: {len(pt_file_data)} bytes")
        
        print(f"📝 Text: {text[:50]}...")
        print(f"🎤 Voice ID: {voice_id}")
        print(f"🎭 Voice Name: {voice_name}")
        print(f"🎵 Audio data present: {audio_data is not None}")
        print(f"📦 PT file data present: {pt_file_data is not None}")
        
        if not text:
            return {"error": "No text provided for synthesis"}
        
        # 🆕 PRIORITY 1: Try .pt file method (if available)
        pt_file_success = False  # 🔧 CRITICAL FIX: Add success flag
        
        if pt_file_data and voice_id != 'default':
            print("🎯 ATTEMPTING PT FILE VOICE LOADING...")
            print(f"📦 PT file for voice: {voice_id}")
            
            try:
                # Convert array back to bytes
                pt_bytes = bytes(pt_file_data)
                print(f"📦 PT file size: {len(pt_bytes)} bytes")
                
                # Load the voice conditionals
                pt_buffer = io.BytesIO(pt_bytes)
                voice_conditionals = torch.load(
                    pt_buffer, 
                    map_location=tts_model.device,
                    weights_only=False  # 🚨 ADDED THIS LINE
                )
                print(f"✅ Successfully loaded voice conditionals from .pt file")
                            
                # Apply voice conditionals to model
                original_conds = getattr(tts_model, 'conds', None)  # Backup original
                tts_model.conds = voice_conditionals
                print(f"🎭 Applied voice conditionals for: {voice_id}")
                
                # Generate with pre-loaded voice
                wav = tts_model.generate(
                    text=text,
                    exaggeration=exaggeration,
                    # NO audio_prompt_path - using pre-loaded conditionals
                )
                
                print("✅ PT FILE VOICE GENERATION SUCCESSFUL!")
                voice_cloning_used = True
                voice_method = "pt_file"
                pt_file_success = True  # 🔧 CRITICAL FIX: Mark as successful
                
                # Restore original conditionals (cleanup)
                tts_model.conds = original_conds
                
            except Exception as e:
                print(f"⚠️ PT file method failed: {str(e)}")
                print(f"📋 PT file error traceback: {traceback.format_exc()}")
                print("🔄 Falling back to original audio method...")
                
                # Reset model state
                tts_model.conds = getattr(tts_model, 'conds', None)
                pt_file_success = False  # 🔧 CRITICAL FIX: Mark as failed
        
        # 🎤 PRIORITY 2: Original audio cloning method (UNCHANGED)
        if not pt_file_success:  # 🔧 CRITICAL FIX: Only fallback if PT file failed
            # EXACT COPY of your existing logic - NO CHANGES
            is_default_voice = (
                not audio_data or 
                audio_data == "" or 
                voice_id == "default" or 
                voice_name.lower() == "default" or
                (audio_data and len(audio_data.strip()) < 100)  # Safe check for None
            )
            
            if is_default_voice:
                print("🔊 DEFAULT VOICE REQUEST DETECTED")
                print("🔊 Using built-in default voice (no voice cloning)")
                
                # Generate with default voice - NO audio_prompt_path
                wav = tts_model.generate(
                    text=text,
                    exaggeration=exaggeration,
                    # NO audio_prompt_path = uses default voice
                )
                
                voice_cloning_used = False
                voice_method = "default"
                
            else:
                print("🎤 LIVE VOICE CLONING REQUEST DETECTED")
                
                # Check if we have audio data for voice cloning
                audio_prompt_path = None
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
                    
                    # Resample if needed
                    if sr != tts_model.sr:
                        print(f"🔄 Resampling from {sr}Hz to {tts_model.sr}Hz")
                        resampler = ta.transforms.Resample(sr, tts_model.sr)
                        voice_sample = resampler(voice_sample)
                        # Save resampled audio back to temp file
                        ta.save(audio_prompt_path, voice_sample, tts_model.sr)
                    
                except Exception as e:
                    print(f"⚠️ Voice cloning setup failed: {str(e)}")
                    print(f"📋 Traceback: {traceback.format_exc()}")
                    print("🔄 Falling back to default voice...")
                    if audio_prompt_path and os.path.exists(audio_prompt_path):
                        os.unlink(audio_prompt_path)
                    audio_prompt_path = None
                
                print("🎵 Generating speech...")
                
                # Generate speech with ChatterboxTTS
                try:
                    if audio_prompt_path:
                        print("🎙️ Generating with voice cloning...")
                        print(f"🎤 Using audio_prompt_path: {audio_prompt_path}")
                        
                        # ✅ CORRECT: Use audio_prompt_path parameter
                        wav = tts_model.generate(
                            text=text,
                            audio_prompt_path=audio_prompt_path,  # ← This enables voice cloning!
                            exaggeration=exaggeration,
                        )
                        print("✅ Voice cloning generation successful!")
                        voice_cloning_used = True
                        voice_method = "live_audio"
                        
                        # Clean up temp file
                        os.unlink(audio_prompt_path)
                        print("🗑️ Cleaned up temp file")
                        
                    else:
                        print("🎤 Voice cloning failed, using default voice...")
                        wav = tts_model.generate(
                            text=text,
                            exaggeration=exaggeration,
                            # NO audio_prompt_path = uses default voice
                        )
                        voice_cloning_used = False
                        voice_method = "default_fallback"
                    
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
        
        # Convert to base64 audio (UNCHANGED)
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
        print(f"🎤 Voice cloning used: {voice_cloning_used}")
        print(f"🔧 Method used: {voice_method}")
        
        # UNCHANGED response format + new method info
        return {
            "audio_base64": audio_b64,
            "sample_rate": tts_model.sr,
            "duration_seconds": round(duration, 2),
            "text": text,
            "voice_cloning_used": voice_cloning_used,
            "voice_id": voice_id,
            "voice_name": voice_name,
            "method_used": voice_method,  # NEW: Shows which method was used
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
