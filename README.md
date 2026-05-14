# MediAug
This is the code repository for the paper:
> **MediAug: Exploring Visual Augmentation in Medical Imaging**
>
> CHEN HONGDA DC229862
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>#</sup>Corresponding author.
>
> ***May 2026***
>
>


## Introduction
Deep learning models for medical image analysis often suffer from performance degradation when tested on data from different domains (e.g., different hospitals, scanners, or patient populations) due to domain shift. Single Domain Generalization (SDG) aims to train a model using data from only one source domain that can generalise well to unseen target domains. Among various SDG techniques, data augmentation is the simplest and most effective approach, as it artificially expands the diversity of the training domain without requiring additional data collection.
Data augmentation is essential in medical imaging for improving classification accuracy, lesion detection, and organ segmentation under limited data conditions. However, two significant challenges remain. First, a pronounced domain gap between natural photographs and medical images can distort critical disease features. Second, augmentation studies in medical imaging are fragmented and limited to single tasks or architectures, leaving the benefits of advanced mix based strategies unclear.
This study systematically evaluates three data augmentation methods – Noise injection, SalfMix (saliency based region mixing), and Elastic transformation – on two medical image datasets (brain tumour MRI and ocular disease fundus) using two backbone networks (ResNet 50 and ViT B). All models were trained on a Google Colab T4 GPU (50 epochs for ViT B, 1 epoch for ResNet 50), and performance was measured using accuracy, F1 score, and ROC AUC.
The experimental results show that Noise augmentation is the most versatile method, achieving the best overall performance for ViT B on both datasets and for ResNet 50 on the ocular disease dataset. SalfMix achieves the highest ROC AUC (98.16%) on the brain tumour dataset with ViT B, demonstrating its strength in fine grained classification of visually similar classes. Elastic transformation performs best for ResNet 50 on the brain tumour dataset (80.75% accuracy) but suffers from low recall (62.28%), indicating a trade off. ViT B consistently outperforms ResNet 50 across all settings, with accuracy gains of 10–18%, suggesting that global self attention is particularly beneficial for medical images with complex spatial structures.

code will be available at https://github.com/hongdachen691-commits/Single-Domain-Generalized-Medical-Image-Analysis



## 🔧 Installation & Setup


To use on **Google Colab** or **Kaggle**, enable GPU and configure data mounting as required.

---

## 📁 Dataset

We use four publicly available medical imaging datasets hosted on Kaggle. In our experiments, the datasets were manually uploaded to Google Drive and accessed through Google Colab notebooks, where all training and evaluation were performed with GPU support.

### 🗂️ Dataset Folder Structure

```
📁 dataset/
├── 📁 brain/
│   ├── Training/
│   ├── Salfmix Training/
│   ├── Noise Training/
│   ├── Elastic Traiing/
│       ├── Training/
│       │   ├── glioma_tumor/
│       │   ├── meningioma_tumor/
│       │   ├── no_tumor/
│       │   └── pituitary_tumor/
│       └── Testing/
│           ├── glioma_tumor/
│           ├── meningioma_tumor/
│           ├── no_tumor/
│           └── pituitary_tumor/
├── 📁 eye/
│   ├── Training/
│   ├── Salfmix Training/
│   ├── Noise Training/
│   ├── Elastic Traiing/
│       ├── Training/
│       │   ├── cataract/
│       │   ├── diabetic_retinopathy/
│       │   ├── glaucoma/
│       │   └── normal/
│       └── Testing/
│           ├── cataract/
│           ├── diabetic_retinopathy/
│           ├── glaucoma/
│           └── normal/
```


### 🧿 Eye Diseases Classification (RGB)

* **URL**: [https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
* Classes: Cataract, Diabetic Retinopathy, Glaucoma, Normal
* Balanced dataset

<p align="left">
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/eye_disease.jpg" width="28%" />
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/eye_tsne.jpg" width="33%" />
</p>

The left pie chart shows the class distribution across the four categories, demonstrating good class balance. The right t-SNE plot provides a feature-level visualization of the high-dimensional distribution of eye disease samples after dimensionality reduction.

### 🧠 Brain Tumor MRI Classification (Grayscale)

* **URL**: [https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data)
* Classes: Glioma, Meningioma, Pituitary, No Tumor
* Imbalanced dataset

<p align="left">
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/brain_disease.jpg" width="28%" />
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/brain_tsne.jpg" width="33%" />
</p>

The pie chart (left) illustrates the class distribution among four tumor categories. The t-SNE plot (right) visualizes the distribution of brain tumor samples in a two-dimensional space, reflecting their separability and overlap in feature space.


## 🏗️ Method Overview

We evaluate six mix-based visual augmentation techniques:

* `MixUp`: Interpolation between image-label pairs
* `YOCO`: Patch-based diverse local/global transforms
* `CropMix`: Multi-scale random crop blending
* `CutMix`: Box-replace image regions + interpolated labels
* `AugMix`: Diverse chained augmentations with consistency
* `SnapMix`: CAM-based semantic-aware mixing

Each method is evaluated on two backbones:

* **ResNet-50** (CNN)
* **ViT-B** (Transformer)


## 💻 Training & Evaluation

To run an experiment with MediAug, follow these steps:

1. **Choose dataset**: `eye` or `brain`
2. **Select model**: `resnet50` or `vit_b`
3. **Pick augmentation method**: one of `salfmix`, `elastic`, `noise`


Training details:

* Epochs: 50
* Optimizer: Adam
* Learning Rate: 0.001
* Batch Size: 32
* Image Size: 224×224
* GPU: Tesla T4 or A100 (Google Colab, via mounted Google Drive)
* CPU: Intel Xeon, 80GB RAM

> **Note:** All experiments were conducted on Google Colab. The datasets were uploaded to Google Drive and accessed using standard Colab notebook mounts (e.g., `from google.colab import drive`). Kaggle was not used for runtime.

* Epochs: 50
* Optimizer: Adam
* Learning Rate: 0.001
* Image Size: 224x224
* Hardware: Tesla T4 / A100, Intel Xeon CPU, 80GB RAM



## 📓 Notebooks

The following notebooks train and evaluate models used in our experiments:

* `resnet50.ipynb`: Trains a ResNet-50 model on the selected dataset with different augmentation strategies. 
* `VIT-B.ipynb`: Trains a ViT-B (Vision Transformer) model on the selected dataset and compares augmentation effects. 
