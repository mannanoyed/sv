FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model during build
RUN mkdir -p pretrained_models/spkrec-ecapa-voxceleb && \
    python -c "\
from speechbrain.pretrained import EncoderClassifier; \
import os; \
print('Downloading model...'); \
classifier = EncoderClassifier.from_hparams( \
    source='speechbrain/spkrec-ecapa-voxceleb', \
    savedir='pretrained_models/spkrec-ecapa-voxceleb' \
); \
print('Model downloaded successfully') \
"

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]