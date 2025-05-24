#!/bin/bash

# Set the GPU you want to use (e.g., GPU 0)
export CUDA_VISIBLE_DEVICES=2,3

# disable the warning for tokenizers
export TOKENIZERS_PARALLELISM=false

# Path to your Python interpreter, if not in default PATH
# PYTHON_EXE=/path/to/your/conda/env/bin/python
PYTHON_EXE=python

# Run the training script
echo "Starting Qwen3 WLAES fine-tuning..."
$PYTHON_EXE train.py # config.yaml is loaded by default within the script

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training script completed successfully."
else
    echo "Training script failed. Check logs for details."
    exit 1
fi

# Optionally, run evaluation after training
# echo "Starting evaluation..."
# $PYTHON_EXE evaluate.py
# if [ $? -eq 0 ]; then
#   echo "Evaluation script completed successfully."
# else
#   echo "Evaluation script failed. Check logs for details."
# fi

echo "Process finished."
