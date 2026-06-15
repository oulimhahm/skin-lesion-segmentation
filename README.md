# 피부 병변 분할과 불확실성 추정

TESL-Net과 Monte Carlo Dropout을 활용한 불확실성 인식 피부 병변 분할 시스템

---

## 개요

흑색종(Melanoma)은 전 세계 피부암 사망의 75%를 차지하지만, 조기 발견 시 5년 생존율이 98% 이상으로 크게 향상된다. 기존 딥러닝 분할 모델은 단일 결정론적 예측만을 출력하며 모델 신뢰도를 정량화하지 않아, 잘못된 예측에 높은 신뢰도를 부여할 경우 임상적으로 위험한 상황을 초래할 수 있다.

본 연구는 분할 마스크와 보정된 불확실성 히트맵을 동시에 출력하는 불확실성 인식 분할 시스템을 제안하여, 임상의가 모호한 케이스를 전문의 검토 대상으로 우선순위를 지정할 수 있도록 지원한다.

---

## 파이프라인

입력 이미지 → OpenCV 전처리 → U-Net / Attention U-Net / TESL-Net + MC Dropout → 분할 마스크 + 불확실성 히트맵

---

## 주요 특징

- DullRazor 알고리즘 기반 체모 제거 (Black-hat 필터링 + INPAINT_TELEA 인페인팅)
- LAB 색 공간의 L 채널에 CLAHE 적용한 조명 보정
- LAB 색 공간 정규화를 통한 색상 표준화
- 세 가지 백본 아키텍처 비교: U-Net, Attention U-Net, TESL-Net (CNN + Swin Transformer + Bi-ConvLSTM)
- MC Dropout (T=50회 확률적 순전파) 기반 불확실성 추정
- 병변 경계부 불확실성 히트맵 시각화
- 동일 학습 조건에서의 세 아키텍처 비교 실험

---

## 실험 결과

ISIC 2018 Task 1 데이터셋 기준.

| 모델 | Dice | IoU | ECE |
|------|------|-----|-----|
| U-Net + MC Dropout | 0.8855 | 0.7964 | 0.0245 |
| Attention U-Net + MC Dropout | 0.8821 | 0.7911 | 0.0186 |
| TESL-Net + MC Dropout | 0.8738 | 0.7777 | 0.0250 |

핵심 발견: TESL-Net의 불확실성 분산값이 U-Net 대비 약 7배 낮으며, 불확실성이 병변 경계부에 정밀하게 집중된다. Dice 수치는 다소 낮으나, 불확실성의 공간적 정밀도 측면에서 임상적으로 명확한 우위를 보여 전문의 검토가 필요한 영역을 직접 지목하는 데 효과적이다.

---

## 불확실성 추정 원리

MC Dropout은 추론 시에도 Dropout을 활성화(`model.train()` 모드 유지, BatchNorm Freeze)하여 T=50회 확률적 순전파를 수행한다.

```
평균 (분할 마스크):   y_bar = (1/T) * sum(y_t)
분산 (불확실성 맵):   sigma^2 = (1/T) * sum((y_t - y_bar)^2)
```

sigma^2 >= 0.01인 케이스는 자동으로 전문의 검토 대상으로 분류된다. (Triage 메커니즘)

---

## 데이터셋

ISIC 2018 Task 1 - 피부 병변 분할

라이센스 제한으로 데이터셋은 본 저장소에 포함되지 않는다. 아래 공식 사이트에서 직접 다운로드:

https://challenge.isic-archive.com/data/

- 훈련: 2,334장
- 검증: 260장
- 입력 해상도: 256 x 256

---

## 저장소 구조

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

## 실험 환경

- Google Colab (NVIDIA T4 GPU)
- Python 3.10
- PyTorch 2.x
- OpenCV 4.x
- timm (Swin Transformer)

---

## 참고 문헌

1. Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI (2015)
2. Gal, Y., Ghahramani, Z.: Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. ICML (2016)
3. Codella, N., et al.: Skin Lesion Analysis Toward Melanoma Detection 2018. arXiv:1902.03368 (2019)
4. Nair, T., et al.: Exploring Uncertainty Measures in Deep Networks for Multiple Sclerosis Lesion Detection and Segmentation. Medical Image Analysis (2020)
5. Iqbal, S., et al.: TESL-Net: A Transformer-Enhanced CNN for Accurate Skin Lesion Segmentation. arXiv:2408.09687 (2024)
