#!/bin/bash
set -e

# Function to preload the Whisper model if needed
preload_whisper_model() {
  WHISPER_MODEL_PATH="/root/.cache/whisper/base.pt"
  WHISPER_MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879326a2df679e5e388781c03a87c20da8bab2c3f781d4feb4/base.pt"
  
  if [ ! -f "$WHISPER_MODEL_PATH" ]; then
    echo "Downloading Whisper base model to cache..."
    mkdir -p /root/.cache/whisper
    # Try with wget first
    if ! wget -q --timeout=30 --tries=3 -O "$WHISPER_MODEL_PATH" "$WHISPER_MODEL_URL"; then
      # If wget fails, try with curl
      echo "wget failed, trying curl..."
      if ! curl -s -o "$WHISPER_MODEL_PATH" "$WHISPER_MODEL_URL"; then
        echo "WARNING: Failed to download Whisper model. It will be downloaded when first used."
      else
        echo "Whisper model successfully downloaded with curl."
      fi
    else
      echo "Whisper model successfully downloaded with wget."
    fi
  else
    echo "Whisper model already exists in cache."
  fi
}

# Wait for Weaviate to be ready
wait_for_weaviate() {
  WEAVIATE_HOST=${WEAVIATE_HOST:-weaviate}
  WEAVIATE_PORT=${WEAVIATE_PORT:-8080}
  WEAVIATE_URL="http://${WEAVIATE_HOST}:${WEAVIATE_PORT}"
  
  echo "Waiting for Weaviate to be available at ${WEAVIATE_URL}..."
  
  # First make sure the host is reachable
  for i in {1..30}; do
    if curl -s --max-time 2 "${WEAVIATE_URL}/v1/meta" > /dev/null; then
      echo "Weaviate is accessible!"
      return 0
    fi
    echo "Waiting for Weaviate server... (attempt $i/30)"
    sleep 2
  done
  
  echo "WARNING: Could not reach Weaviate after 30 attempts, but will proceed anyway."
  echo "The application may not function correctly until Weaviate is available."
  return 0  # Return success anyway so the container doesn't exit
}

# Print diagnostic information
print_diagnostics() {
  echo "Environment variables:"
  echo "WEAVIATE_HOST=${WEAVIATE_HOST:-weaviate}"
  echo "WEAVIATE_PORT=${WEAVIATE_PORT:-8080}"
  
  echo "Network connectivity check:"
  ping -c 1 ${WEAVIATE_HOST:-weaviate} || echo "Cannot ping Weaviate host (this is normal if ICMP is blocked)"
  
  echo "DNS resolution:"
  getent hosts ${WEAVIATE_HOST:-weaviate} || echo "Cannot resolve Weaviate host via DNS"
  
  echo "Container network information:"
  ip addr show || echo "Cannot show network interfaces"
}

# Run initialization steps
echo "Starting initialization..."
preload_whisper_model
print_diagnostics
wait_for_weaviate

# Start the Streamlit app with error handling
echo "Starting Streamlit application..."
streamlit run main.py --server.port=8501 --server.address=0.0.0.0 || {
  echo "Streamlit application failed to start. Printing error information:"
  print_diagnostics
  # Sleep to keep container running so logs can be inspected
  echo "Container will remain running for 1 hour to allow log inspection."
  sleep 3600
}