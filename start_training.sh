#!/bin/bash
# Activate the virtual environment
source ./venv/bin/activate
# Run the training script with some default parameters
python scripts/train.py --epochs 5 --lr 0.0001 --model_file_name "my_caption_model"
# Deactivate the virtual environment
deactivate
