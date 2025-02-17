#!/bin/bash

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null 
then
    echo "git-lfs is not installed. Please install git-lfs before running this script."
    exit 1
fi

# Define the Hugging Face repository URLs for the model and dataset
MODEL_REPO="https://huggingface.co/code-philia/CoEdPilot-generator"
DATASET_REPO="https://huggingface.co/datasets/code-philia/CoEdPilot-generator"

# Create directories to save the model and dataset
mkdir -p model
mkdir -p dataset

# Clone the model repository into the model directory
echo "Downloading the model..."
git lfs clone $MODEL_REPO model
if [ $? -ne 0 ]; then
    echo "Model download failed. Please check your network or the repository URL."
    exit 1
fi

# Clone the dataset repository into the dataset directory
echo "Downloading the dataset..."
git lfs clone $DATASET_REPO dataset
if [ $? -ne 0 ]; then
    echo "Dataset download failed. Please check your network or the repository URL."
    exit 1
fi

echo "Model and dataset download completed."