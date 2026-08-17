# Boundary-Aware Fast-SCNN

This repository provides an optimized PyTorch implementation of Fast-SCNN, tailored specifically for agricultural lane and plot segmentation. The core enhancement is the integration of the **Boundary Intersection over Union (BIoU)** loss mechanism, which significantly improves boundary prediction accuracy.

## 📌 Background and Motivation

In agricultural environments, traditional semantic segmentation models often struggle to capture precise boundaries between navigable lanes and crop plots. Small boundary deviations can lead to significant navigation errors for autonomous vehicles. 

To address this, we integrated the **Boundary-aware Loss (BIoU)** into the training phase. Unlike standard Cross-Entropy or OHEM losses that treat all pixels equally, our Boundary-aware Loss explicitly penalizes misclassifications near the semantic boundaries. 
- **Optimization Impact**: By using morphological operations (dilation and erosion) to extract boundaries dynamically, the model is forced to focus its learning capacity on difficult topological edges. This results in much cleaner predicted masks and vastly reduces steering/navigation offsets in real-world deployments.

## 🏗️ Underlying Framework
This project is built upon the Fast-SCNN architecture.
- **Paper**: [Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/abs/1902.04502)
- **Original PyTorch Implementation**: [Fast-SCNN-pytorch](https://github.com/Tramac/Fast-SCNN-pytorch)

## 📁 Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Boundary-Aware-Fast-SCNN.git
   cd Boundary-Aware-Fast-SCNN
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start (Demo)

This repository comes with a set of default demo files located in `datasets/` and `weights/`. You can immediately test the pipeline without any configuration.

### 1. Training Test
Run a quick test training loop on the demo dataset.
```bash
python train.py --epochs 1 --batch-size 1
```

### 2. Evaluation
Evaluate the model and automatically generate `pred_result/evaluation_metrics.csv` containing MIoU and BIoU metrics for each image.
```bash
python eval.py
```

### 3. Inference (Single Image)
Predict the mask for a single image and save the result to `pred_result/`.
```bash
python predict.py
```

## ⚙️ Configuration
The `config.yaml` file controls the entire training and dataset logic.
- `use_biou`: Set to `true` to enable Boundary-aware loss, `false` for standard loss.
- `dataset_type`: Supports `custom_agricultural` (2 classes), `asparagus` (3 classes), and `public_cityscapes` (19 classes).

## 📊 Evaluation Metrics
During `eval.py`, regardless of whether `use_biou` is turned on for training, the script will strictly calculate both **MIoU** and **BIoU** for every single sample. The granular results will be saved in a CSV format for detailed analysis.

## 🖼️ Assets & Images
*If you wish to add images or GIFs to this README, it is highly recommended to place them in an `assets/` or `pictures/` directory to keep the root directory clean.*
