# Base image with PyTorch + CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime


# Install matching PyTorch packages
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121


# Install system dependencies
RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1-dev \
    && apt-get clean

# Set workdir
WORKDIR /app

# Copy your Python code and other necessary files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# If you're using local encoders or models, ensure they are copied above
# and your script points to the correct paths

# Default command to run your script (edit as needed)
CMD ["python", "run_gradio.py"]

