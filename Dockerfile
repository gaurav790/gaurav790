# Healthcare AI - Chest X-ray Pneumonia Detection
# Production-ready Docker container with CPU/GPU support

# Use official PyTorch image as base (includes CUDA support)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for production
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn[standard] \
    fastapi \
    python-multipart \
    aiofiles

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data \
    /app/models \
    /app/logs \
    /app/checkpoints \
    /app/evaluation_results \
    /app/uploads \
    /app/outputs

# Set permissions
RUN chmod +x /app/*.py

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('Health check passed')" || exit 1

# Expose ports
EXPOSE 8000 8501 6006

# Default command
CMD ["python", "train.py", "--help"]

# Multi-stage build for smaller production image (optional)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as production

WORKDIR /app

# Copy only necessary files from build stage
COPY --from=0 /app/requirements.txt .
COPY --from=0 /app/*.py .
COPY --from=0 /app/config.yaml .

# Install minimal dependencies for production
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/data /app/models /app/logs /app/outputs

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Production command
CMD ["python", "inference.py", "--help"]