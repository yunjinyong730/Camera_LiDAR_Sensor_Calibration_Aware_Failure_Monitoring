# CalibGuard-TF-Pro

<p align="center">
  <b>Camera–LiDAR Calibration Drift Detection & Recovery Monitor</b><br/>
  TensorFlow 기반 경량 센서 보정 모델로 카메라–라이다 extrinsic drift를 감지하고, 6DoF 보정값을 예측하며, 복구 여부를 시각화하는 프로젝트입니다.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange"/>
  <img src="https://img.shields.io/badge/Task-Camera--LiDAR%20Calibration-green"/>
  <img src="https://img.shields.io/badge/Model-MSF--CalibNet-purple"/>
  <img src="https://img.shields.io/badge/Demo-Recovery%20Report-blue"/>
  <img src="https://img.shields.io/badge/Export-TFLite-lightgrey"/>
</p>

---

## 목차

- [1. 프로젝트 개요](#1-프로젝트-개요)
- [2. 왜 이 프로젝트를 만들었는가](#2-왜-이-프로젝트를-만들었는가)
- [3. 데모 결과 해석](#3-데모-결과-해석)
- [4. 전체 파이프라인](#4-전체-파이프라인)
- [5. 모델 입력과 출력](#5-모델-입력과-출력)
- [6. MSF-CalibNet 구조](#6-msf-calibnet-구조)
- [7. Recovery Mode](#7-recovery-mode)
- [8. Recovery Metric](#8-recovery-metric)
- [9. 프로젝트 구조](#9-프로젝트-구조)
- [10. 실행 방법](#10-실행-방법)
- [11. 실제 KITTI 학습](#11-실제-kitti-학습)
- [12. 결과를 어떻게 봐야 하는가](#12-결과를-어떻게-봐야-하는가)
- [13. 포트폴리오 어필 포인트](#13-포트폴리오-어필-포인트)
- [14. 한계와 다음 단계](#14-한계와-다음-단계)

---

## 1. 프로젝트 개요

**CalibGuard-TF-Pro**는 자율주행/로보틱스 환경에서 발생할 수 있는 **카메라–라이다 센서 보정 오차**를 다루는 프로젝트입니다.

일반적인 perception pipeline은 카메라와 라이다가 정확히 정렬되어 있다고 가정합니다. 하지만 실제 시스템에서는 진동, 온도 변화, 장착 오차, 장기 운용에 따른 drift 때문에 calibration이 틀어질 수 있습니다.

이 프로젝트는 다음 질문에서 출발했습니다.

> 카메라–라이다 extrinsic calibration이 틀어졌을 때, perception/fusion pipeline이 얼마나 무너지는지 감지하고, 이를 다시 복구할 수 있을까?

CalibGuard는 KITTI-style camera–LiDAR 데이터를 이용해 synthetic drift를 주입하고, TensorFlow 기반 경량 모델인 **MSF-CalibNet**으로 6DoF 보정값을 예측합니다. 이후 normal, drifted, recovered projection을 비교하여 실제로 복구가 되었는지 정량적/시각적으로 판단합니다.

---

## 2. 왜 이 프로젝트를 만들었는가

카메라–라이다 센서 융합에서는 LiDAR point를 image plane에 정확히 projection하는 것이 중요합니다. Extrinsic calibration이 조금만 틀어져도 다음 문제가 생길 수 있습니다.

- LiDAR point가 실제 객체 위치가 아닌 다른 image 영역에 projection됨
- camera object와 LiDAR point association이 틀어짐
- depth-aware detection 또는 fusion confidence가 떨어짐
- perception failure가 발생했는데 image-only 결과만 보면 원인을 찾기 어려움

그래서 CalibGuard는 단순히 calibration matrix를 계산하는 프로젝트가 아니라, **calibration error가 perception/fusion 결과에 어떤 failure를 만드는지 보여주는 failure monitor**로 설계했습니다.

핵심 목표는 세 가지입니다.

| 목표 | 설명 |
|---|---|
| Drift simulation | 정상 extrinsic에 의도적으로 6DoF calibration drift를 주입 |
| Learning-based recovery | 경량 TensorFlow 모델이 drift correction을 예측 |
| Recovery visualization | 복구가 되었는지 overlay, error vector, metric으로 확인 |

---

## 3. 데모 결과 해석

### 3.1 Recovery Report

아래 이미지는 한 프레임에서 normal, drifted, recovered projection을 함께 보여주는 report입니다.

<p align="center">
  <img src="assets/recovery_report_model_edge_refined.png" width="95%"/>
</p>

Report에는 다음 정보가 포함됩니다.

| 항목 | 의미 |
|---|---|
| Normal projection | 원래 KITTI calibration으로 projection한 결과 |
| Drifted projection | synthetic drift가 들어간 잘못된 projection |
| Recovered projection | 모델 또는 refinement 이후 복구된 projection |
| Tri-color overlay | normal/drifted/recovered를 한 이미지에 겹쳐 표시 |
| Error vectors | normal 기준 drifted/recovered가 얼마나 벗어났는지 표시 |
| Recovery status | 회복 여부를 수치로 판정 |

### 3.2 Tri-color Overlay

<p align="center">
  <img src="assets/tricolor_overlay_model_edge_refined.png" width="95%"/>
</p>

Overlay 색상은 다음과 같습니다.

| 색상 | 의미 |
|---|---|
| Green | 정상 calibration projection |
| Red | drift가 들어간 projection |
| Cyan | 복구 후 projection |

해석은 단순합니다.

> Cyan 점들이 Red 점들보다 Green 점에 가까워지면 recovery가 된 것입니다.

반대로 Cyan이 Green에서 멀어지거나 Red와 비슷한 위치에 있으면 아직 복구가 충분하지 않은 것입니다.

---

## 4. 전체 파이프라인

CalibGuard의 전체 흐름은 다음과 같습니다.

```text
KITTI image + LiDAR + calibration
        |
        v
정상 projection 생성
        |
        v
Synthetic 6DoF drift 주입
        |
        v
Drifted LiDAR projection 생성
        |
        v
RGB / Depth / Edge / Residual feature 생성
        |
        v
MSF-CalibNet이 6DoF correction 예측
        |
        v
Recovered extrinsic 생성
        |
        v
Normal / Drifted / Recovered 비교
        |
        v
Recovery rate, status, demo video 생성
```

수식으로 표현하면 다음과 같습니다.

```text
T_gt = original KITTI Tr_velo_to_cam
T_bad = Delta(drift) @ T_gt
T_recovered = Delta(predicted_correction) @ T_bad
```

모델은 `T_bad`로 인해 어긋난 projection pattern을 보고 correction을 예측합니다.

```text
MSF-CalibNet(input) -> [roll, pitch, yaw, tx, ty, tz], confidence
```

---

## 5. 모델 입력과 출력

### 5.1 입력

MSF-CalibNet은 raw point cloud를 직접 입력받지 않습니다. 대신 camera image plane 기준으로 정렬된 6채널 입력을 사용합니다.

| 채널 | 설명 |
|---|---|
| RGB image | 카메라 이미지 |
| Sparse depth map | drifted calibration으로 projection한 LiDAR depth |
| Image edge map | 이미지 경계/객체 boundary 정보 |
| Residual map | LiDAR point와 image edge 사이의 alignment residual |

입력 형태 예시:

```text
H x W x 6
```

Toy demo에서는 다음 크기를 사용했습니다.

```text
128 x 384 x 6
```

실제 KITTI 학습에서는 다음 크기를 권장합니다.

```text
192 x 640 x 6
```

### 5.2 출력

모델 출력은 두 개입니다.

| 출력 | 설명 |
|---|---|
| 6DoF correction | `[roll, pitch, yaw, tx, ty, tz]` |
| Confidence | calibration 상태에 대한 confidence score |

6DoF correction은 잘못된 extrinsic을 정상 방향으로 되돌리기 위한 보정값입니다.

---

## 6. MSF-CalibNet 구조

**MSF-CalibNet**은 camera–LiDAR calibration correction을 위해 만든 경량 TensorFlow CNN입니다.

일반적인 image classification CNN이 아니라, 다음 alignment cue를 학습하도록 설계했습니다.

- RGB image structure
- Sparse LiDAR projection
- Image edge
- LiDAR-edge residual
- Projection shift pattern

구성 요소는 다음과 같습니다.

| 모듈 | 역할 |
|---|---|
| Fused Inverted Residual Block | 초기 stage에서 효율적인 feature extraction |
| Depthwise Inverted Residual Block | 연산량을 줄인 spatial feature extraction |
| Strip Depthwise Convolution | 가로/세로 방향 alignment shift pattern 포착 |
| Multi-stage Token Fusion | 여러 scale의 feature를 global token으로 통합 |
| Correction Head | 6DoF extrinsic correction 예측 |
| Confidence Head | calibration confidence 예측 |

구조 요약:

```text
RGB + Drifted Depth + Edge + Residual
            |
      Lightweight CNN Encoder
            |
      Multi-stage Feature Fusion
            |
   -----------------------------
   |                           |
6DoF Correction           Confidence
```

---

## 7. Recovery Mode

데모 스크립트는 여러 recovery mode를 제공합니다.

| Mode | 설명 | 사용 목적 |
|---|---|---|
| `model` | MSF-CalibNet 예측값만 사용 | 순수 learning-based recovery |
| `model_edge_refined` | 모델 예측 후 edge alignment refinement 적용 | 추천 데모 모드 |
| `edge_refined` | 모델 없이 edge 기반 refinement만 수행 | geometry baseline |
| `oracle` | synthetic drift의 정답 inverse 사용 | upper-bound sanity check |

추천 모드는 다음입니다.

```text
model_edge_refined
```

`oracle` 모드는 실제 배포용이 아닙니다. 이 모드는 projection, metric, visualization pipeline이 정상인지 확인하기 위한 상한선 실험입니다.

---

## 8. Recovery Metric

Recovered가 실제로 좋아졌는지 눈으로만 판단하면 애매합니다. 그래서 CalibGuard는 같은 원본 LiDAR point index를 기준으로 reprojection error를 계산합니다.

| Metric | 의미 |
|---|---|
| Drifted error | normal projection과 drifted projection 사이의 평균 pixel error |
| Recovered error | normal projection과 recovered projection 사이의 평균 pixel error |
| Recovery rate | drifted error 대비 recovered error가 얼마나 줄었는지 |
| Recovery status | 회복 여부 판정 |

Recovery rate는 다음과 같이 계산합니다.

```text
recovery_rate = (drifted_error - recovered_error) / drifted_error * 100
```

판정 기준은 다음과 같습니다.

| Status | 기준 |
|---|---|
| `RECOVERED` | recovery rate >= 30% |
| `PARTIALLY_RECOVERED` | recovery rate >= 5% |
| `NOT_RECOVERED` | recovery rate < 5% |

예를 들어:

```text
Drifted error   = 19.72 px
Recovered error = 24.53 px
Recovery rate   = -24.4%
Status          = NOT_RECOVERED
```

이 경우 recovered projection이 normal에 가까워진 것이 아니라 더 멀어졌기 때문에 복구 실패로 판단합니다.

---

## 9. 프로젝트 구조

```text
CalibGuard-TF-Pro/
  calibguard/
    data/
      kitti_io.py
      dataset.py

    geometry/
      se3.py
      projection.py

    features/
      input_builder.py

    models/
      msf_calibnet.py
      serialization.py

    recovery/
      edge_refine.py

    metrics/
      calibration_metrics.py

    utils/
      visualization.py
      profiler.py
      tflite.py

  scripts/
    00_make_toy_kitti.py
    01_check_projection.py
    02_train.py
    03_evaluate.py
    04_demo_recovery.py
    05_export_tflite.py
    06_streamlit_dashboard.py
    07_make_demo_video.py

  assets/
    recovery_report_model_edge_refined.png
    tricolor_overlay_model_edge_refined.png
    calibguard_demo.mp4
```

각 폴더 역할:

| 폴더 | 역할 |
|---|---|
| `data` | KITTI 파일 로딩과 drift dataset 생성 |
| `geometry` | SE(3), projection, reprojection error 계산 |
| `features` | RGB/depth/edge/residual input 생성 |
| `models` | TensorFlow MSF-CalibNet 구현 |
| `recovery` | edge-based refinement |
| `metrics` | rotation/translation/reprojection/recovery metric |
| `utils` | visualization, profiler, TFLite export |
| `scripts` | 학습, 평가, 데모, 비디오 생성 entrypoint |

---

## 10. 실행 방법

### 10.1 환경 설치

```powershell
conda env create -f environment.yml
conda activate calibguard-tf
python -m pip install -e .
```

TensorFlow와 NumPy 버전 확인:

```powershell
python -c "import sys, numpy as np, tensorflow as tf; print(sys.executable); print('numpy', np.__version__); print('tensorflow', tf.__version__)"
```

권장 버전:

```text
Python      3.11
NumPy       1.26.4
TensorFlow  2.15.x
```

TensorFlow가 NumPy 2.x와 충돌하면 다음을 실행합니다.

```powershell
python -m pip install "numpy==1.26.4"
```

---

### 10.2 Toy KITTI 생성

```powershell
python scripts/00_make_toy_kitti.py --out_dir data/toy_kitti/training --num_frames 24
```

### 10.3 Projection 확인

```powershell
python scripts/01_check_projection.py --data_root data/toy_kitti/training --frame_id 000000 --out_dir outputs/toy_projection_check
```

### 10.4 Toy 데이터 학습

```powershell
python scripts/02_train.py --data_root data/toy_kitti/training --out_dir runs/toy_msf_calibnet --epochs 50 --batch_size 4 --steps_per_epoch 100 --val_steps 20 --img_h 128 --img_w 384
```

### 10.5 평가

```powershell
python scripts/03_evaluate.py --data_root data/toy_kitti/training --model_path runs/toy_msf_calibnet/best.keras --out_csv runs/toy_msf_calibnet/eval.csv --num_samples 100 --img_h 128 --img_w 384
```

### 10.6 Recovery demo 생성

```powershell
python scripts/04_demo_recovery.py --data_root data/toy_kitti/training --model_path runs/toy_msf_calibnet/best.keras --frame_id 000000 --out_dir outputs/toy_recovery_demo --img_h 128 --img_w 384 --recovery_mode model_edge_refined
```

### 10.7 Demo video 생성

```powershell
python scripts/07_make_demo_video.py --data_root data/toy_kitti/training --model_path runs/toy_msf_calibnet/best.keras --out_dir outputs/toy_demo_video --num_frames 24 --fps 4 --img_h 128 --img_w 384 --recovery_mode model_edge_refined --ramp_drift
```

---

## 11. 실제 KITTI 학습

Toy KITTI는 pipeline 검증용입니다. 포트폴리오 결과로 사용하려면 실제 KITTI Object Detection 데이터를 아래 구조로 준비하는 것을 권장합니다.

```text
data/
  kitti_object/
    training/
      image_2/
      velodyne/
      calib/
      label_2/
```

### 11.1 Projection 확인

```powershell
python scripts/01_check_projection.py --data_root data/kitti_object/training --frame_id 000000 --out_dir outputs/kitti_projection_check
```

### 11.2 Real training

```powershell
python scripts/02_train.py --data_root data/kitti_object/training --out_dir runs/kitti_msf_calibnet_real --epochs 50 --batch_size 8 --steps_per_epoch 1000 --val_steps 150 --img_h 192 --img_w 640
```

### 11.3 Strong training

```powershell
python scripts/02_train.py --data_root data/kitti_object/training --out_dir runs/kitti_msf_calibnet_strong --epochs 80 --batch_size 8 --steps_per_epoch 1500 --val_steps 200 --img_h 192 --img_w 640
```

### 11.4 평가

```powershell
python scripts/03_evaluate.py --data_root data/kitti_object/training --model_path runs/kitti_msf_calibnet_real/best.keras --out_csv runs/kitti_msf_calibnet_real/eval.csv --num_samples 500 --img_h 192 --img_w 640
```

### 11.5 Demo report

```powershell
python scripts/04_demo_recovery.py --data_root data/kitti_object/training --model_path runs/kitti_msf_calibnet_real/best.keras --frame_id 000000 --out_dir outputs/kitti_recovery_demo_real --img_h 192 --img_w 640 --recovery_mode model_edge_refined
```

### 11.6 Demo video

```powershell
python scripts/07_make_demo_video.py --data_root data/kitti_object/training --model_path runs/kitti_msf_calibnet_real/best.keras --out_dir outputs/kitti_demo_video_real --num_frames 48 --fps 4 --img_h 192 --img_w 640 --recovery_mode model_edge_refined --ramp_drift
```

---

## 12. 결과를 어떻게 봐야 하는가

### 12.1 정상적으로 회복된 경우

정상적으로 회복되면 다음 현상이 나타납니다.

- Cyan 점이 Red 점보다 Green 점에 가까워짐
- Error vector에서 cyan line이 red line보다 짧아짐
- Recovered error가 Drifted error보다 작아짐
- Recovery rate가 양수로 증가
- Status가 `RECOVERED` 또는 `PARTIALLY_RECOVERED`

### 12.2 회복되지 않은 경우

다음과 같으면 아직 모델이 calibration correction을 제대로 학습하지 못한 것입니다.

- Cyan 점이 Green과 멀리 있음
- Recovered error가 Drifted error보다 큼
- Recovery rate가 음수
- Status가 `NOT_RECOVERED`

이 경우는 visualization 문제가 아니라, 모델 학습이 부족하거나 toy data가 충분하지 않은 상황일 수 있습니다.

현재 toy KITTI 기반 짧은 학습에서는 `NOT_RECOVERED`가 나올 수 있습니다. 이 프로젝트에서 중요한 것은 pipeline이 다음을 모두 지원한다는 점입니다.

```text
drift 주입
model correction 예측
projection 복구
동일 LiDAR point 기반 error 계산
recovery status 판정
자동 report/video 생성
```

---

## 13. 포트폴리오 어필 포인트

이 프로젝트는 단순히 LiDAR를 이미지에 얹는 projection demo가 아닙니다.

다음 내용을 보여줍니다.

| 역량 | 프로젝트에서 보여주는 부분 |
|---|---|
| Sensor calibration | Camera–LiDAR extrinsic drift와 6DoF correction |
| Perception geometry | LiDAR-to-image projection, reprojection error |
| Sensor fusion understanding | Camera image와 LiDAR point alignment 분석 |
| Lightweight deep learning | TensorFlow 기반 MSF-CalibNet |
| Edge AI awareness | TFLite export |
| Failure monitoring | Recovery rate, status, confidence |
| Visualization | Tri-color overlay, error vector, automatic video |

면접에서 설명할 때는 다음 문장을 사용할 수 있습니다.

> 저는 KITTI-style camera–LiDAR 데이터를 이용해 synthetic extrinsic drift를 생성하고, TensorFlow 기반 경량 모델이 6DoF calibration correction을 예측하도록 학습했습니다. 또한 recovered projection이 normal projection으로 실제로 가까워졌는지 동일 LiDAR point 기준 reprojection error, tri-color overlay, error vector, recovery rate로 검증했습니다.

---

## 14. 한계와 다음 단계

현재 버전의 한계는 다음과 같습니다.

- Toy KITTI는 pipeline 검증용이며 최종 성능 평가용이 아님
- 현재 모델은 single-frame 기반임
- Temporal drift trend는 아직 반영하지 않음
- Object-level IoU / association metric은 추가 가능
- ROS2/C++ 실시간 integration은 다음 단계

향후 개선 방향:

- [ ] 실제 KITTI Object Detection 전체 학습
- [ ] TESLA-style temporal encoder 추가
- [ ] KITTI Tracking 기반 temporal calibration drift detection
- [ ] projected 3D box IoU metric 추가
- [ ] object association failure rate 추가
- [ ] ROS2 node 추가
- [ ] C++ inference stub 추가
- [ ] TFLite latency benchmark 추가

---

## Citation-style Summary

```bibtex
@misc{calibguard_tf_pro,
  title  = {CalibGuard-TF-Pro: Calibration-Aware Camera-LiDAR Perception Failure Monitor},
  author = {Your Name},
  year   = {2026},
  note   = {TensorFlow implementation for camera-LiDAR extrinsic drift detection and recovery}
}
```
