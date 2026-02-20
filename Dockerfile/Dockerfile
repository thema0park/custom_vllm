FROM nvidia/cuda:12.0.0-base-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git vim htop wget \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio && \
    pip install --no-cache-dir transformers datasets peft bitsandbytes pandas numpy datasets

WORKDIR /home/jhs/projects/personal-multi-agent