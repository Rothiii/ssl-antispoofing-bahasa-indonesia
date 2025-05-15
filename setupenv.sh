#!/bin/bash

# Make sure you have anaconda installed and available in your PATH

set -e

read -r -p "Enter environment name (default: ssl): " ENV_NAME
ENV_NAME=${ENV_NAME:-ssl}

echo "📦 Creating conda environment: $ENV_NAME with Python 3.7"
conda create -y -n "$ENV_NAME" python=3.7

echo "✅ Environment $ENV_NAME successfully created"

echo "🚀 Activating environment $ENV_NAME"
conda activate "$ENV_NAME"

echo "⚙️ Installing PyTorch 1.8.1 + CUDA 11.1"
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

echo "🧠 Installing fairseq (commit a540213)"
pip install git+https://github.com/pytorch/fairseq.git@a54021305d6b3c4c5959ac9395135f63202db8f1

echo "📄 Installing all dependencies from requirements.txt"
pip install -r requirements.txt

echo "🎉 Setup complete! Environment '$ENV_NAME' is ready to use."
