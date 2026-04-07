# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings('ignore')
import os
import random
import numpy as np
import torch
from ultralytics import YOLO


# Function to set random seed
def set_seed(seed=42):
    """Set all random seeds to ensure reproducibility"""
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch CPU random seed

    # If using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Current GPU
        torch.cuda.manual_seed_all(seed)  # All GPUs
        torch.backends.cudnn.deterministic = True  # Ensure deterministic convolution operations
        torch.backends.cudnn.benchmark = False  # Disable cuDNN auto-tuner

    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


if __name__ == '__main__':
    # Set random seed (42 is recommended, but can be customized)
    SEED = 42
    set_seed(SEED)

    # Create model
    model = YOLO(model=r'D:\Yolov13\yolov13\ultralytics\cfg\models\v13\yolov13s.yaml')

    # Load pretrained weights (enable as needed)
    model.load('yolov13s.pt')   # yolov13n.pt

    # Training configuration
    model.train(
        pretrained=True,
        data=r'data.yaml',
        imgsz=640,
        epochs=500,
        patience=50,
        batch=24,
        workers=0,
        device='',
        optimizer='AdamW', # 'SGD' 'Adam' 'AdamW'
        close_mosaic=False,
        resume=False,
        project='runs/train',
        name='exp',
        single_cls=False,
        cache=False,
        lr0=0.00015, # Initial learning rate
        #lrf=0.01,  # Final learning rate factor = lr0 * lrf (0.002 * 0.01 = 0.00002)
        cos_lr = True, # Use cosine annealing scheduler

        # Data augmentation settings
        augment=False,
        auto_augment = False,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        erasing = 0.0,
        copy_paste = False,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees = 0.0,
        translate = 0.0,
        scale= 0.0,
        shear= 0.0,

        # Loss weights
        cls = 0.5,
        box = 7.5,
        dfl = 1.5,

        # Key settings for reproducibility
        deterministic=True,  # Enable deterministic mode
        seed=SEED,  # Explicitly pass seed to training process
    )
