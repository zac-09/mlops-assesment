# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure we have the model files (they should be copied with the application code)
# Download weights only if not present
RUN if [ ! -f "pytorch_model_weights.pth" ]; then \
    echo "Downloading model weights..."; \
    python download_weights.py; \
    else \
    echo "Model weights already present"; \
    fi

# Convert to ONNX only if not present  
RUN if [ ! -f "model.onnx" ]; then \
    echo "Converting model to ONNX..."; \
    python convert_to_onnx.py; \
    else \
    echo "ONNX model already present"; \
    fi

# Expose port for Cerebrium
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]