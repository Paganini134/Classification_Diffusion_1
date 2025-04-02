# Classification_Diffusion_1

# Chest Tumor Detection Using Diffusion Models

A research-driven project aimed at detecting chest tumors using high-resolution medical imaging datasets. This approach leverages **Diffusion Models** and has shown to outperform conventional CNN-based architectures like ResNet-18, ResNet-50, and GoogleNet.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Performance Graphs](#performance-graphs)
6. [Ablation Studies](#ablation-studies)
7. [How to Use](#how-to-use)
8. [Project Structure](#project-structure)
9. [License](#license)
10. [Contact](#contact)

---

## Introduction

This repository contains an **academic project** (August 2024 – Present) under the guidance of **Prof. Vinay Chamola**, focusing on building a **Diffusion-based Classification Model** for chest tumor detection. Our goal is to provide an innovative and efficient solution for medical image analysis, improving accuracy and reducing error rates compared to traditional CNN-based approaches.

---

## Key Features

- **High-Resolution Medical Datasets**: Utilizes large and high-quality datasets for robust training and evaluation.
- **Diffusion-Based Classification**: Applies diffusion models for classification tasks, outperforming standard CNNs (ResNet-18, ResNet-50, GoogleNet).
- **Ablation Studies**: Investigates the impact of different scheduling methods (linear vs. cosine) and timestep embeddings on model performance.
- **Advanced Architectures**: Experiments with U-Net and multi-head attention mechanisms to further boost accuracy and lower loss.
- **Superior Performance**: Achieved a **95.83% accuracy** and an **MSE loss of 0.027**, significantly better than conventional CNN-based approaches.

---

## Methodology

1. **Data Collection & Preprocessing**  
   - Acquired high-resolution medical images from publicly available datasets.  
   - Normalized and standardized images to ensure consistency across the dataset.

2. **Model Architecture**  
   - **Diffusion Model**: Trained using forward and reverse diffusion processes for classification.  
   - **U-Net & Attention**: Integrated multi-head attention blocks into the U-Net backbone to capture intricate features.  
   - **Scheduling & Embeddings**: Conducted ablation studies on linear vs. cosine scheduling and different timestep embeddings.

3. **Training**  
   - Loss functions used: **Cross-Entropy** for classification and **MSE** for reconstruction checks.  
   - Optimized with **Adam/AdamW** (configurable) and tested across multiple epochs to ensure convergence.

4. **Evaluation**  
   - Compared with **ResNet-18**, **ResNet-50**, and **GoogleNet** on the same dataset.  
   - Measured metrics: **Accuracy**, **MSE Loss**, **Precision**, **Recall**, and **F1 Score**.

---

## Results

- **Accuracy**: 95.83%  
- **MSE Loss**: 0.027  
- **Outperformed** ResNet-18, ResNet-50, and GoogleNet in both accuracy and MSE loss metrics.

---

## Performance Graphs

Below are the main performance graphs. Make sure you update the image filenames or paths accordingly:

### Accuracy
![Accuracy Graph](images/accuracy.png)

### Negative Log-Likelihood (NLL)
![NLL Graph](images/nll.png)

### Mean Squared Error (MSE)
![MSE Graph](images/mse.png)

---

## Ablation Studies

We conducted extensive ablation studies to understand the impact of scheduling methods, timestep embeddings, and various architectural tweaks.

Below are placeholders for your ablation graphs. Update filenames/paths as needed:

![Ablation Graph 1](images/ablation1.png)
![Ablation Graph 2](images/ablation2.png)
![Ablation Graph 3](images/ablation3.png)
![Ablation Graph 4](images/ablation4.png)
![Ablation Graph 5](images/ablation5.png)
![Ablation Graph 6](images/ablation6.png)

---

## How to Use

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/chest-tumor-detection-diffusion.git
   cd chest-tumor-detection-diffusion
