# Introduction
A detection-promptable segmentation framework (CoreDetSeg) that integrates YOLOv13 and SAM for automated drill core lithology classification and RQD estimation.

# License
Our work builds upon the YOLO13 and Segment Anything Model (SAM) implementations from Ultralytics, which are licensed under the GNU Affero General Public License v3.0 and the Apache 2.0 License, respectively. Our project code is released under the MIT License.

# Installation
1. First, obtain the correct PyTorch installation command for your system from the official PyTorch website.
2. Then, install the remaining dependencies for this project by running the following command:
```
pip install -r requirements.txt
```

# Usage
## Dataset structure
### Dats arrangement
Organize your original images and LabelMe JSON annotations in this structure:
```
raw_dataset/
├── images/
│   ├── core_box_001.jpg
│   ├── core_box_002.jpg
│   ├── core_box_003.jpg
│   └── ...
└── annotations/
    ├── core_box_001.json
    ├── core_box_002.json
    ├── core_box_003.json
    └── ...
```
### Dataset in YOLO format
After running **preprocess_mydataset.py**, your dataset will be organized in the YOLO format, as below:
```
dataset/
├── images/
│   ├── train/           # Training images (60% of total)
│   │   ├── core_box_001.jpg
│   │   ├── core_box_002.jpg
│   │   └── ...
│   ├── val/             # Validation images (20% of total)
│   │   ├── core_box_101.jpg
│   │   ├── core_box_102.jpg
│   │   └── ...
│   └── test/            # Test images (20% of total)
│       ├── core_box_201.jpg
│       ├── core_box_202.jpg
│       └── ...
└── labels/
    ├── train/           # Training labels
    │   ├── core_box_001.txt
    │   ├── core_box_002.txt
    │   └── ...
    ├── val/             # Validation labels
    │   ├── core_box_101.txt
    │   ├── core_box_102.txt
    │   └── ...
    └── test/            # Test labels
        ├── core_box_201.txt
        ├── core_box_202.txt
        └── ...
```
## Dataset preparation
To convert JSON annotations (e.g., from LabelMe) to YOLO format and split the dataset into train/val/test sets, modify the paths in **preprocess_mydataset.py** and run the following command:
```
python preprocess_mydataset.py
```
## Model training
To train the YOLOv13 on your core box dataset, first updata the **data.yaml** configuration file with your dataset paths. Then, modify the parameters in **train_mydataset.py** and run the following command: 
```
python train_mydataset.py
```
## Model ouput and evaluation
The **test_metrcis.py** script evaluates the trained YOLO11 performance on test datasets. It calculates four evaluation metrics including precision, recall, mAP@50, and mAP@50-95 for quantitative model assessment. To evaluate the performance of the trained YOLO11 on unseen dataset, first updata the **test.yaml** configuration file with your dataset paths. Then, modify the parameters in **test_metrcis.py** and run the following command: 
```
python test_metrcis.py
```

## RQD Result Visualization
The **rqd_calculation.py** script performs RQD analysis on core box images using YOLO11-SAM. It detects core segments, calculates RQD values along multiple scanlines, and provides an interactive visualization interface with real-time parameter adjustment. To perform RQD analysis on a single core box image, first update the configuration parameters in **rqd_calculation.py** with your model paths, image path, and output directory. Then run the following command:
```
python rqd_calculation.py
```

# Citations and acknowledgements
This project is built upon the following foundational works. Please cite them if you use our code:
## **AutoRQD:**
"A Zero-Shot Segmentation Framework with Detection Prompts for Automated Rock Quality Designation (RQD) Estimation from Core Box Images" (https://doi.org/10.1016/j.asoc.2026.114886)
## **YOLO13:**
@article{yolov13,
  title={YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception},
  author={Lei, Mengqi and Li, Siqi and Wu, Yihong and et al.},
  journal={arXiv preprint arXiv:2506.17733},
  year={2025}
}
## **SAM:**
@misc{kirillov2023segment,
      title={Segment Anything},
      author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
      year={2023},
      eprint={2304.02643},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
