#!/bin/bash

# Activate the conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./env # Use conda activate for newer versions of Conda

# Set CUDA_VISIBLE_DEVICES to 0
echo "Setting CUDA_VISIBLE_DEVICES to 0..."
export CUDA_VISIBLE_DEVICES=0

# Confirming environment setup
echo "Conda environment activated and CUDA_VISIBLE_DEVICES set to 0."

# Print the name of the GPU
echo "Checking GPU availability..."
python -c "import torch; print(torch.cuda.get_device_name(0))"
echo "This should be the name of the GPU you are using."
