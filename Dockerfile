FROM python:3.12

WORKDIR /app

# Install system dependencies required for PyTorch, ffmpeg (audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data persistence and Whisper cache
RUN mkdir -p /root/.cache/whisper audio_files transcripts

# Copy the startup script
COPY ./src ./src
COPY start.sh .
RUN chmod +x start.sh

# Copy application files
COPY *.py ./

# Expose port for Streamlit
EXPOSE 8501

# Create volumes for persistent data
VOLUME ["/app/audio_files", "/app/transcripts"]

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Use the startup script as entrypoint
CMD ["./start.sh"]
