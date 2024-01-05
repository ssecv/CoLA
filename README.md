
# Project Title

## Environment Requirements
- Python 3.8
- PyTorch 1.9
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

- `options.py`: Configuration file where all the settings for training and testing are modified.
- `ResNet.py`: Contains ResNet-related backbone models and configurations.
- `test.py`: The script used for testing the models.
- `train.py`: The script used for training the models.
- `pytorch_iou`: Contains code for IoU (Intersection over Union) loss computation.
- `data.py`: Includes operations related to data loading and processing.
- `clip`: Files related to the CLIP model.
- `Net.py`: The main network architecture.

