# Skin Lesion Segmentation with Uncertainty Estimation

Early melanoma detection system using U-Net + MC Dropout for skin lesion segmentation and uncertainty visualization.

## Overview

Skin cancer (melanoma) accounts for 75% of skin cancer deaths worldwide. Early detection raises survival rate to over 98%, but visual diagnosis by dermatologists achieves only 75~84% accuracy. This project builds an AI-assisted diagnostic system that outputs both a segmentation mask and an uncertainty heatmap to help clinicians prioritize cases for review.

## Pipeline

Input Image → OpenCV Preprocessing → U-Net + MC Dropout → Segmentation Mask + Uncertainty Heatmap

## Key Features

- Hair removal using DullRazor algorithm (OpenCV inpainting)
- Lighting correction using CLAHE
- Color normalization using LAB color space
- U-Net encoder-decoder architecture with skip connections
- MC Dropout (T=50 iterations) for uncertainty estimation
- Uncertainty heatmap visualization highlighting ambiguous regions
- Attention U-Net comparison experiment

## Results

| Model | Dice Coefficient | IoU Score | ECE |
|---|---|---|---|
| U-Net + MC Dropout | 0.8876 | 0.7992 | 0.0185 |
| Attention U-Net + MC Dropout | 0.8842 | - | - |

## Dataset

ISIC 2018 Task 1 - Skin Lesion Segmentation
- Train: 2,334 images
- Validation: 260 images
- Image size: 256 x 256

## Environment

- Google Colab (NVIDIA T4 GPU)
- Python 3.10
- PyTorch 2.x
- OpenCV 4.x

## References

- Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015
- Gal & Ghahramani, Dropout as a Bayesian Approximation, ICML 2016
- Codella et al., Skin Lesion Analysis Toward Melanoma Detection 2018, arXiv 2019
