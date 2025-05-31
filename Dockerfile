# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies including git (needed for some packages)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies with better error handling
# Try to install flash-attn separately and continue if it fails
RUN pip install --no-cache-dir torch>=2.0.0 torchaudio>=2.0.0 || echo "Failed to install torch, continuing..."
RUN pip install --no-cache-dir transformers>=4.30.0 accelerate>=0.20.0 || echo "Failed to install transformers, continuing..."
RUN pip install --no-cache-dir runpod>=1.7.0 || echo "Failed to install runpod, continuing..."

# Try to install flash-attn separately (optional, will continue if fails)
RUN pip install --no-cache-dir flash-attn>=2.0.0 || echo "flash-attn failed to install, continuing without it..."

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt || echo "Some packages failed to install, continuing..."

# Install Chatterbox TTS specifically
RUN pip install --no-cache-dir chatterbox-tts || echo "Failed to install chatterbox-tts, continuing..."

# Copy handler code
COPY handler.py /app/handler.py

# Copy test input (optional)
COPY test_input.json /app/test_input.json 2>/dev/null || echo "No test input file found"

# Create directory for model cache
RUN mkdir -p /app/models

# Set environment variables for model caching
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_DATASETS_CACHE=/app/models

# Expose port (though RunPod handles this automatically)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import runpod; print('RunPod imported successfully')" || exit 1

# Command to run the handler
CMD ["python", "-u", "handler.py"]
