# Troubleshooting Guide

This document provides solutions for common issues you might encounter when running the YouTube RAG application.

## Connection Issues Between Containers

### Symptoms
- The RAG application container can't connect to Weaviate
- You see log messages like "Waiting for Weaviate..." repeatedly
- The application exits with code 1 after timeout

### Solutions

1. **Check if Weaviate is running properly**
   ```bash
   docker-compose ps
   ```
   Confirm that both services are running. If Weaviate shows "unhealthy" status, check its logs:
   ```bash
   docker-compose logs weaviate
   ```

2. **Check network connectivity**
   Make sure both containers are on the same network:
   ```bash
   docker network inspect app_rag-network
   ```
   You should see both containers listed under "Containers".

3. **Verify Weaviate API readiness**
   You can manually check if the Weaviate API is responding:
   ```bash
   curl http://localhost:8080/v1/.well-known/ready
   ```
   This should return `{"ready":true}`.

4. **Restart with proper dependency ordering**
   The updated docker-compose file includes health checks to ensure Weaviate is ready before starting the RAG app:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

## Disk Space Issues

### Symptoms
- You see warnings about disk usage in Weaviate logs
- Operations fail due to insufficient disk space
- Database operations are slow

### Solutions

1. **Clean up Docker volumes and cache**
   Use the provided cleanup script:
   ```bash
   chmod +x volume_cleanup.sh
   ./volume_cleanup.sh
   ```

2. **Start fresh if needed**
   If problems persist, you can remove the Weaviate data directory and start fresh:
   ```bash
   docker-compose down
   rm -rf ./weaviate-data
   docker-compose up -d
   ```

3. **Allocate more disk space to Docker**
   - On Docker Desktop, you can increase disk space in Settings > Resources > Disk Image Size
   - On Linux hosts, make sure the partition where Docker stores data has sufficient free space

## Model Download Issues

### Symptoms
- Whisper model download fails
- Application can't transcribe audio

### Solutions

1. **Manual model download**
   Download the model manually and place it in the correct folder:
   ```bash
   mkdir -p ~/.cache/whisper
   curl -o ~/.cache/whisper/base.pt https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879326a2df679e5e388781c03a87c20da8bab2c3f781d4feb4/base.pt
   ```
   Then map this directory to the container:
   ```yaml
   volumes:
     - ~/.cache/whisper:/root/.cache/whisper
   ```

2. **Check internet connectivity**
   Ensure the container has internet access to download the model:
   ```bash
   docker-compose exec rag-app curl -I https://openaipublic.azureedge.net
   ```

## API Keys and Authentication

### Symptoms
- Application fails with authentication errors
- Embedding or LLM calls fail

### Solutions

1. **Verify environment variables**
   Make sure your `.env` file exists and contains all required API keys:
   ```bash
   cat .env
   ```
   You should see all three required keys:
   ```
   OPENAI_API_KEY=sk-...
   HF_SERVERLESS_INFERENCE_TOKEN=hf_...
   ELEVENLABS_API_KEY=...
   ```

2. **Test API key validity**
   Test each API key separately:
   ```bash
   # OpenAI
   curl -s https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY" | jq

   # HuggingFace (check HTTP status)
   curl -s -o /dev/null -w "%{http_code}" \
     -H "Authorization: Bearer $HF_SERVERLESS_INFERENCE_TOKEN" \
     https://api-inference.huggingface.co/status

   # ElevenLabs (check HTTP status)
   curl -s -o /dev/null -w "%{http_code}" \
     -H "xi-api-key: $ELEVENLABS_API_KEY" \
     https://api.elevenlabs.io/v1/user
   ```

## Performance Issues

### Symptoms
- Transcription is very slow
- Vector searches take a long time
- Application becomes unresponsive

### Solutions

1. **Check resource usage**
   ```bash
   docker stats
   ```
   
2. **Reduce chunk size for transcripts**
   Modify `TranscriptionProcessor` in `get_transcriptions.py` to use smaller chunks:
   ```python
   def __init__(self, chunk_size=2000, chunk_overlap=200):
   ```

3. **Use a lighter Whisper model**
   Change the Whisper model in `AudioTranscriber` to a smaller size:
   ```python
   def __init__(self, model_name="tiny", transcripts_path="transcripts"):
   ```

4. **Increase Docker resource limits**
   Allocate more CPU/memory to Docker in Docker Desktop settings or through container resource limits.