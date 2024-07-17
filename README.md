# [ECCV 2024] CoLA: Conditional Dropout and Language-driven Robust Dual-modal Salient Object Detection
This repository hosts the code of the research presented in "CoLA: Conditional Dropout and Language-driven Robust Dual-modal Salient Object Detection". We provide both **PyTorch** and **MindSpore** versions of the code.

![intro](img/model.png)
## Environment Requirements
- Python 3.8
- PyTorch 2.1
- torch-npu 2.1.0
- torchvision

## Installation
To set up the necessary environment, follow these steps:

```bash
pip install torch==1.9 torchvision
```

## Training
To train the model, run the following command:

```bash
python train.py
```

Configuration settings can be adjusted in `options.py`. This file contains various parameters and settings that you can modify to customize the training process.

## Testing
For testing the model, use the following command:

```bash
python test.py
```

Similar to training, the testing configurations can be found and altered in `options.py`.


## File Structure
This repository is organized into two primary directories to accommodate both **PyTorch** and **MindSpore** codebases, ensuring compatibility and ease of use across different deep learning frameworks. Each directory mirrors the following structure:
- `options.py`: Configuration file where all the settings for training and testing are modified.
- `ResNet.py`: Contains ResNet-related backbone models and configurations.
- `test.py`: The script used for testing the models.
- `train.py`: The script used for training the models.
- `pytorch_iou`: Contains code for IoU (Intersection over Union) loss computation.
- `data.py`: Includes operations related to data loading and processing.
- `clip`: Files related to the CLIP model.
- `Net.py`: The main network architecture.

