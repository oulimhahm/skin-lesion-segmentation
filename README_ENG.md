# Skin Lesion Segmentation with Uncertainty Estimation

Uncertainty-aware skin lesion segmentation system for early melanoma detection using TESL-Net and Monte Carlo Dropout.

---

## Overview

Melanoma accounts for 75% of skin cancer deaths worldwide, yet early detection raises the 5-year survival rate to over 98%. Current deep learning segmentation models produce only deterministic predictions without quantifying model confidence, which poses a risk in clinical settings where overconfident wrong predictions can mislead clinicians.

This project proposes an uncertainty-aware segmentation system that simultaneously outputs a segmentation mask and a calibrated uncertainty heatmap, enabling clinicians to prioritize ambiguous cases for expert review.

---

## Pipeline

Input Image → OpenCV Preprocessing → U-Net / Attention U-Net / TESL-Net + MC Dropout → Segmentation Mask + Uncertainty Heatmap

---

## Key Features

- Hair removal using DullRazor algorithm (Black-hat filtering + INPAINT_TELEA inpainting)
- Lighting correction via CLAHE applied to the L channel of LAB color space
- Color normalization using LAB color space standardization
- Three backbone architectures: U-Net, Attention U-Net, TESL-Net (CNN + Swin Transformer + Bi-ConvLSTM)
- MC Dropout (T=50 stochastic forward passes) for uncertainty estimation
- Uncertainty heatmap visualization highlighting ambiguous boundary regions
- Comparative experiment across three architectures under identical training conditions

---

## Results

Evaluated on ISIC 2018 Task 1 dataset.

| Model | Dice | IoU | ECE |
|-------|------|-----|-----|
| U-Net + MC Dropout | 0.8855 | 0.7964 | 0.0245 |
| Attention U-Net + MC Dropout | 0.8821 | 0.7911 | 0.0186 |
| TESL-Net + MC Dropout | 0.8738 | 0.7777 | 0.0250 |

Key finding: TESL-Net achieves approximately 7x lower uncertainty variance compared to U-Net, with uncertainty concentrated precisely at lesion boundaries. This spatial precision in uncertainty estimation offers a clear clinical advantage for identifying regions requiring expert review, despite slightly lower Dice scores.

---

## Uncertainty Estimation

MC Dropout maintains Dropout active during inference (`model.train()` mode with BatchNorm frozen) and performs T=50 stochastic forward passes.

```
Mean (segmentation mask):   y_bar = (1/T) * sum(y_t)
Variance (uncertainty map): sigma^2 = (1/T) * sum((y_t - y_bar)^2)
```

Cases with sigma^2 >= 0.01 are flagged for specialist review (Triage mechanism).

---

## Dataset

ISIC 2018 Task 1 - Skin Lesion Segmentation

The dataset is not included in this repository due to licensing restrictions. Please download directly from the official source:

https://challenge.isic-archive.com/data/

- Train: 2,334 images
- Validation: 260 images
- Image size: 256 x 256

---

## Repository Structure

```
skin-lesion-segmentation/
├── README.md
├── README_KO.md
├── skin_lesion_segmentation.ipynb                 
└── results/
    ├── training_curves.png
    ├── model_comparison.png
    ├── uncertainty_visualization.png
    ├── tesl_uncertainty_visualization.png
    ├── tesl_learning_curve.png
    └── calibration_curve.png
```

---

## Environment

- Google Colab (NVIDIA T4 GPU)
- Python 3.10
- PyTorch 2.x
- OpenCV 4.x
- timm (Swin Transformer)

---

## References

1. Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI (2015)
2. Gal, Y., Ghahramani, Z.: Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. ICML (2016)
3. Codella, N., et al.: Skin Lesion Analysis Toward Melanoma Detection 2018. arXiv:1902.03368 (2019)
4. Nair, T., et al.: Exploring Uncertainty Measures in Deep Networks for Multiple Sclerosis Lesion Detection and Segmentation. Medical Image Analysis (2020)
5. Iqbal, S., et al.: TESL-Net: A Transformer-Enhanced CNN for Accurate Skin Lesion Segmentation. arXiv:2408.09687 (2024)
