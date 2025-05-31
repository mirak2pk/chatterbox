import io
import os
import base64
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import uvicorn
import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from chatterbox.tts import ChatterboxTTS

# Global variable to store the model
model = None

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech", max_length=5000)
    exaggeration: float = Field(0.5, description="Emotion exaggeration level", ge=0.0, le=1.0)
    cfg_weight: float = Field(0.5, description="CFG weight for generation", ge=0.0, le=1.0)
    temperature: float = Field(0.7, description="Temperature for generation", ge=0.1, le=2.0)
    return_base64: bool = Field(False, description="Return audio as base64 string instead of streaming")

class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration_seconds: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    print("Loading Chatterbox TTS model...")
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("Using CPU")
    
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    yield
    
    # Shutdown
    print("Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Chatterbox TTS API",
    description="Text-to-Speech API using Resemble AI's Chatterbox model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this to your iOS app's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Chatterbox TTS API", "status": "running"}

@app.get("/health")
async def health_check():
    global model
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "device": str(next(model.t3.parameters()).device) if model else "unknown"
    }

@app.post("/generate", response_model=TTSResponse)
async def generate_speech_base64(request: TTSRequest):
    """Generate speech and return as base64 encoded audio"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate audio
        with torch.inference_mode():
            wav = model.generate(
                text=request.text,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                temperature=request.temperature
            )
        
        # Convert to bytes
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        
        # Calculate duration
        duration = wav.shape[-1] / model.sr
        
        # Encode to base64
        audio_bytes = buffer.getvalue()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return TTSResponse(
            audio_base64=audio_base64,
            sample_rate=model.sr,
            duration_seconds=round(duration, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.post("/generate/stream")
async def generate_speech_stream(
    text: str = Form(...),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    temperature: float = Form(0.7)
):
    """Generate speech and return as streaming audio file"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate audio
        with torch.inference_mode():
            wav = model.generate(
                text=text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
        
        # Convert to bytes
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=generated_speech.wav"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.post("/generate/voice-clone")
async def generate_speech_with_voice_clone(
    text: str = Form(...),
    audio_file: UploadFile = File(...),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    temperature: float = Form(0.7),
    return_base64: bool = Form(False)
):
    """Generate speech with voice cloning from uploaded audio file"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate audio file
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{audio_file.filename}"
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Generate audio with voice cloning
        with torch.inference_mode():
            wav = model.generate(
                text=text,
                audio_prompt_path=temp_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Convert to bytes
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        
        if return_base64:
            # Calculate duration and return base64
            duration = wav.shape[-1] / model.sr
            audio_bytes = buffer.getvalue()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            return TTSResponse(
                audio_base64=audio_base64,
                sample_rate=model.sr,
                duration_seconds=round(duration, 2)
            )
        else:
            # Return streaming response
            return StreamingResponse(
                io.BytesIO(buffer.read()),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=voice_cloned_speech.wav"}
            )
        
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

def handler(event):
    """RunPod serverless handler function"""
    try:
        # Extract input from event
        input_data = event.get("input", {})
        text = input_data.get("text", "")
        exaggeration = input_data.get("exaggeration", 0.5)
        cfg_weight = input_data.get("cfg_weight", 0.5)
        temperature = input_data.get("temperature", 0.7)
        
        if not text:
            return {"error": "Text input is required"}
        
        # Generate audio
        with torch.inference_mode():
            wav = model.generate(
                text=text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
        
        # Convert to base64
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        
        audio_bytes = buffer.getvalue()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        duration = wav.shape[-1] / model.sr
        
        return {
            "audio_base64": audio_base64,
            "sample_rate": model.sr,
            "duration_seconds": round(duration, 2)
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Check if running in RunPod serverless environment
    if os.getenv("RUNPOD_ENDPOINT_ID"):
        # RunPod serverless mode
        import runpod
        runpod.serverless.start({"handler": handler})
    else:
        # Local development mode
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            workers=1
        )