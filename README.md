# CalibGuard-TF Pro

**CalibGuard-TF Pro** is a TensorFlow camera-LiDAR calibration recovery project for KITTI-style data.

This version upgrades the previous project in three important ways:

1. **Clear recovery judgement**: it now computes `recovery_rate_percent` and `recovery_status` so you can tell whether recovery worked.
2. **Better visualization**: it saves a report image with normal/drifted/recovered projections, tri-color overlay, error vectors, and metric bars.
3. **Automatic demo video**: it creates an MP4 video for README, GitHub, or interview demos.

Color convention in the enhanced visualizations:

```text
Green = Normal projection using original KITTI calibration
Red   = Drifted projection after synthetic calibration error
Cyan  = Recovered projection after model/refinement correction
```

If recovery works, **cyan should move closer to green than red**. The metric panel also reports:

```text
Recovery rate = (Drifted reprojection error - Recovered reprojection error) / Drifted error * 100
```

Status rule:

```text
RECOVERED             recovery_rate >= 30%
PARTIALLY_RECOVERED   recovery_rate >= 5%
NOT_RECOVERED         recovery_rate < 5%
```

---

## Windows PowerShell quick commands

Use one-line commands in PowerShell. Do not use Linux `\` line continuation.

### 1. Environment

```powershell
conda deactivate
conda env remove -n calibguard-tf -y
conda create -n calibguard-tf python=3.11 -y
conda activate calibguard-tf
python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy==1.26.4" "tensorflow==2.15.1" "opencv-python==4.9.0.80" "pyyaml" "tqdm" "pandas" "matplotlib" "streamlit" "scikit-learn"
python -m pip install -e .
```

### 2. Make toy KITTI data

```powershell
python scripts/00_make_toy_kitti.py --out_dir data/toy_kitti/training --num_frames 24
```

### 3. Check projection

```powershell
python scripts/01_check_projection.py --data_root data/toy_kitti/training --frame_id 000000 --out_dir outputs/projection_check
```

### 4. Train quick toy model

```powershell
python scripts/02_train.py --data_root data/toy_kitti/training --out_dir runs/toy_msf_calibnet --epochs 2 --batch_size 2 --steps_per_epoch 5 --val_steps 2 --img_h 128 --img_w 384
```

> Note: a 2-epoch toy model is only a smoke test. It may not recover well. For a visually guaranteed sanity check, run demo with `--recovery_mode oracle`. For a more practical demo, use `--recovery_mode model_edge_refined` after longer training.

### 5. Enhanced one-frame recovery report

```powershell
python scripts/04_demo_recovery.py --data_root data/toy_kitti/training --model_path runs/toy_msf_calibnet/best.keras --frame_id 000000 --out_dir outputs/recovery_demo --img_h 128 --img_w 384 --recovery_mode model_edge_refined
```

Open the generated report:

```powershell
explorer outputs\recovery_demo
```

Main file:

```text
outputs/recovery_demo/000000_recovery_report.png
```

### 6. Automatic demo video

```powershell
python scripts/07_make_demo_video.py --data_root data/toy_kitti/training --model_path runs/toy_msf_calibnet/best.keras --out_dir outputs/demo_video --num_frames 12 --fps 3 --img_h 128 --img_w 384 --recovery_mode model_edge_refined --ramp_drift
```

Open the video folder:

```powershell
explorer outputs\demo_video
```

Main file:

```text
outputs/demo_video/calibguard_demo.mp4
```

---

## Recovery modes

`04_demo_recovery.py` and `07_make_demo_video.py` support four recovery modes.

```text
model               model prediction only
model_edge_refined  model prediction + small edge-alignment local search
edge_refined         edge-alignment local search from zero correction
oracle              exact inverse synthetic drift; sanity-check only
```

Use `oracle` only to verify that the projection/recovery visualization works. It is not a deployable method because it uses the known synthetic drift.

Use `model_edge_refined` for the main portfolio demo. It shows the learned calibration model plus a lightweight geometric refinement step.

---

## Why the previous demo looked unclear

The old demo showed three separate images:

```text
Normal | Drifted | Recovered
```

That made it hard to judge whether recovery worked because LiDAR projections are sparse and visually noisy.

This Pro version fixes that with:

1. **same-point correspondence**: normal, drifted, and recovered points are compared using the same original LiDAR point indices.
2. **tri-color overlay**: green/red/cyan projections are drawn on one image.
3. **error vectors**: red line = drift error, cyan line = recovered error.
4. **metric panel**: recovery rate and status are shown directly.
5. **demo video**: recovery is shown across multiple frames or drift magnitudes.

---

## Important honesty note

A tiny toy dataset and 2 training epochs are only for checking that the code runs. The model may output weak corrections, so `NOT_RECOVERED` can be a valid result. That is not a visualization bug; it means the trained model has not learned enough yet.

For real experiments, use KITTI Object Detection data and train longer, for example:

```powershell
python scripts/02_train.py --data_root data/kitti_object/training --out_dir runs/kitti_msf_calibnet --epochs 30 --batch_size 8 --steps_per_epoch 1000 --val_steps 100 --img_h 192 --img_w 640
```

Then run:

```powershell
python scripts/07_make_demo_video.py --data_root data/kitti_object/training --model_path runs/kitti_msf_calibnet/best.keras --out_dir outputs/kitti_demo_video --num_frames 24 --fps 4 --img_h 192 --img_w 640 --recovery_mode model_edge_refined --ramp_drift
```

---


# CalibGuard-TF

**CalibGuard-TF**는 KITTI camera-LiDAR 데이터를 이용해 **extrinsic calibration drift**를 합성하고, TensorFlow 기반 경량 CNN 모델인 **MSF-CalibNet**으로 6DoF 보정값을 학습·예측하는 프로젝트입니다.

이 프로젝트의 최종 데모 흐름은 다음과 같습니다.

```text
KITTI image + Velodyne LiDAR + calibration
→ synthetic extrinsic drift injection
→ RGB / sparse depth / edge / residual feature 생성
→ MSF-CalibNet이 6DoF correction + confidence 예측
→ recovered extrinsic으로 재투영
→ 동일 LiDAR point index 기준 drifted vs recovered reprojection error 비교
```

---

## 1. 프로젝트 목표

기존의 단순 camera-LiDAR projection demo가 아니라, 다음을 모두 포함합니다.

1. **KITTI camera-LiDAR projection**
2. **Synthetic calibration drift dataset generation**
3. **TensorFlow 기반 경량 CNN sensor calibration model**
4. **6DoF extrinsic correction regression**
5. **Calibration confidence estimation**
6. **Drifted / recovered projection visualization**
7. **Reprojection error, rotation error, translation error 평가**
8. **TFLite export**
9. **Streamlit dashboard**

이 프로젝트는 센서 보정 연구를 자율주행/로보틱스 perception/fusion 문제로 확장하는 것을 목표로 합니다.

---

## 2. 모델 개요: MSF-CalibNet

**MSF-CalibNet**은 camera-LiDAR alignment 상태를 입력받아 calibration correction을 예측하는 경량 CNN입니다.

### 입력

모델 입력은 한 프레임마다 6채널 tensor입니다.

```text
RGB image                  3 channels
Drifted LiDAR depth map    1 channel
Image edge map             1 channel
LiDAR-edge residual map    1 channel
-----------------------------------
Total                      6 channels
```

### 출력

```text
correction: [roll, pitch, yaw, tx, ty, tz]
confidence: calibration confidence score
```

- `roll, pitch, yaw`: degrees
- `tx, ty, tz`: meters
- 학습 중에는 안정성을 위해 normalized target을 사용합니다. Target correction은 단순 `-drift`가 아니라 synthetic drift의 **SE(3) inverse**로 생성합니다.

### 핵심 구조

```text
Fused Inverted Residual Block
Depthwise Inverted Residual Block
Strip Depthwise Convolution Block
Multi-stage token fusion
Correction head + Confidence head
```

`Strip Depthwise Convolution`은 `1xK`, `Kx1` depthwise convolution을 사용합니다. Camera-LiDAR projection mismatch는 객체 경계 주변에서 수평/수직 shift로 많이 나타나기 때문에, 일반 3x3 CNN보다 alignment pattern을 더 잘 포착하도록 설계했습니다.

---

## 3. Conda 환경 설정

### 3.1 Conda 환경 생성

```bash
conda env create -f environment.yml
conda activate calibguard-tf
```

### 3.2 pip 기반 설치

Conda 환경 생성 후 프로젝트 루트에서 다음을 실행합니다.

```bash
pip install -e .
```

### 3.3 GPU 사용 시

TensorFlow GPU 환경은 CUDA/cuDNN 조합에 따라 달라질 수 있습니다. 우선 CPU 환경에서 smoke test가 정상 동작하는지 확인한 뒤 GPU 환경을 맞추는 것을 추천합니다.

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

---

## 4. 데이터 준비

### 4.1 실제 KITTI Object Detection 데이터

다음 구조를 기대합니다.

```text
data/kitti_object/training/
  image_2/
    000000.png
    000001.png
    ...
  velodyne/
    000000.bin
    000001.bin
    ...
  calib/
    000000.txt
    000001.txt
    ...
  label_2/       # optional
```

본 프로젝트의 핵심 학습은 `image_2`, `velodyne`, `calib`만 있어도 동작합니다. `label_2`는 나중에 object-level metric을 추가할 때 사용합니다.

### 4.2 바로 실행 가능한 toy KITTI 데이터 생성

KITTI를 아직 받지 않았더라도 전체 코드가 돌아가는지 확인할 수 있도록 toy KITTI-like 데이터를 생성하는 스크립트를 포함했습니다.

```bash
python scripts/00_make_toy_kitti.py --out_dir data/toy_kitti/training --num_frames 24
```

생성되는 구조는 실제 KITTI와 동일합니다.

```text
data/toy_kitti/training/
  image_2/
  velodyne/
  calib/
```

Toy 데이터는 실제 성능 평가용이 아니라, 코드 실행과 파이프라인 검증용입니다.

---

## 5. 빠른 실행 순서

### Step 1. Toy 데이터 생성

```bash
python scripts/00_make_toy_kitti.py --out_dir data/toy_kitti/training --num_frames 24
```

### Step 2. Projection 확인

```bash
python scripts/01_check_projection.py \
  --data_root data/toy_kitti/training \
  --frame_id 000000 \
  --out_dir outputs/projection_check
```

출력:

```text
outputs/projection_check/000000_normal_projection.png
outputs/projection_check/000000_drifted_projection.png
```

### Step 3. 짧은 학습 실행

```bash
python scripts/02_train.py \
  --data_root data/toy_kitti/training \
  --out_dir runs/toy_msf_calibnet \
  --epochs 2 \
  --batch_size 2 \
  --steps_per_epoch 5 \
  --val_steps 2 \
  --img_h 128 \
  --img_w 384
```

실제 KITTI에서는 예를 들어 다음처럼 실행합니다.

```bash
python scripts/02_train.py \
  --data_root data/kitti_object/training \
  --out_dir runs/kitti_msf_calibnet \
  --epochs 30 \
  --batch_size 8 \
  --steps_per_epoch 1000 \
  --val_steps 100 \
  --img_h 192 \
  --img_w 640
```

### Step 4. 평가

```bash
python scripts/03_evaluate.py \
  --data_root data/toy_kitti/training \
  --model_path runs/toy_msf_calibnet/best.keras \
  --out_csv runs/toy_msf_calibnet/eval.csv \
  --num_samples 20 \
  --img_h 128 \
  --img_w 384
```

### Step 5. Recovery demo 이미지 생성

```bash
python scripts/04_demo_recovery.py \
  --data_root data/toy_kitti/training \
  --model_path runs/toy_msf_calibnet/best.keras \
  --frame_id 000000 \
  --out_dir outputs/recovery_demo \
  --img_h 128 \
  --img_w 384
```

출력:

```text
outputs/recovery_demo/000000_normal.png
outputs/recovery_demo/000000_drifted.png
outputs/recovery_demo/000000_recovered.png
outputs/recovery_demo/000000_comparison.png
outputs/recovery_demo/000000_metrics.json
```

### Step 6. TFLite export

```bash
python scripts/05_export_tflite.py \
  --data_root data/toy_kitti/training \
  --model_path runs/toy_msf_calibnet/best.keras \
  --out_path runs/toy_msf_calibnet/msf_calibnet_int8.tflite \
  --img_h 128 \
  --img_w 384 \
  --samples 10
```

### Step 7. Streamlit dashboard

```bash
streamlit run scripts/06_streamlit_dashboard.py -- \
  --data_root data/toy_kitti/training \
  --model_path runs/toy_msf_calibnet/best.keras \
  --img_h 128 \
  --img_w 384
```

---

## 6. Config 기반 실행

기본 설정 파일은 `configs/default.yaml`입니다. CLI 인자가 우선 적용되도록 구성했습니다.

```bash
python scripts/02_train.py --config configs/default.yaml
```

---

## 7. 핵심 수식

KITTI LiDAR point를 이미지 평면에 투영할 때 다음 식을 사용합니다.

```text
x_img = P2 · R0_rect · Tr_velo_to_cam · x_velo
u = x_img[0] / x_img[2]
v = x_img[1] / x_img[2]
```

Synthetic drift는 다음과 같이 적용합니다.

```text
T_bad = Delta(drift) · T_gt
```

모델은 correction vector를 예측합니다.

```text
model(input) → Δξ_pred = [roll, pitch, yaw, tx, ty, tz]
T_recovered = Delta(Δξ_pred) · T_bad
```

학습 label은 synthetic drift transform의 정확한 inverse입니다.

```text
target_correction = inverse_SE3(drift)
Delta(target_correction) · T_bad ≈ T_gt
```

최종 평가는 단순 parameter error뿐 아니라 실제 recovered projection이 정상 projection과 가까워지는지까지 확인합니다. Reprojection error는 동일한 원본 LiDAR point index가 두 projection에서 모두 image 안에 있을 때만 계산합니다.

---

## 8. 코드 구조

```text
CalibGuard-TF/
  README.md
  environment.yml
  requirements.txt
  pyproject.toml

  configs/
    default.yaml

  calibguard/
    common/
      config.py
      seed.py

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

    losses/
      calibration_losses.py

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
```

---

## 9. 실험 결과 표 예시

실제 KITTI 학습 후 README에 다음 표를 채우면 됩니다.

| Model | Rot. Error ↓ | Trans. Error ↓ | Reproj. Error ↓ | Params ↓ | Latency ↓ |
|---|---:|---:|---:|---:|---:|
| No Recovery | x° | x m | x px | - | - |
| Grid Search | x° | x m | x px | - | x ms |
| CNN Baseline | x° | x m | x px | x M | x ms |
| **MSF-CalibNet** | x° | x m | x px | x M | x ms |

Recovery demo 표는 다음처럼 구성합니다.

| Scenario | Reprojection Error ↓ | Confidence ↑ |
|---|---:|---:|
| Normal | x px | 0.95 |
| Drifted | x px | 0.xx |
| Recovered | x px | 0.xx |

---

## 10. 주의 사항

1. Toy 데이터는 성능 평가용이 아닙니다. 코드 실행 확인용입니다.
2. 실제 KITTI에서는 `steps_per_epoch`, `epochs`, `img_h`, `img_w`를 충분히 키우세요.
3. Target correction은 `inverse_SE3(drift)`로 생성합니다. 단순 `-drift` 근사는 쓰지 않습니다.
4. 논문/포트폴리오용으로는 반드시 동일 LiDAR point index 기준 recovered projection error를 함께 보고하세요.
5. 이미지 단독 detector mAP는 extrinsic drift에 직접적으로 영향을 받지 않으므로, calibration-aware metric을 사용해야 합니다.

---

## 11. 추천 개발 순서

1. `00_make_toy_kitti.py`로 toy 데이터 생성
2. `01_check_projection.py`로 projection이 정상인지 확인
3. `02_train.py`로 짧은 학습 smoke test
4. 실제 KITTI 데이터로 학습
5. `03_evaluate.py`로 parameter/reprojection error 평가
6. `04_demo_recovery.py`로 portfolio 이미지 생성
7. `05_export_tflite.py`로 경량 배포 가능성 확인
8. `06_streamlit_dashboard.py`로 데모 UI 구성

---

## 12. 포트폴리오 소개 문장

> CalibGuard-TF is a TensorFlow-based calibration-aware perception project that learns camera-LiDAR extrinsic correction from synthetic KITTI calibration drift. It predicts 6DoF correction and calibration confidence using a lightweight multi-stage fusion CNN, then validates recovery through reprojection error and projection visualization.


---

## 11. 구현상 주의점과 수정 사항

이 버전에서는 초기 구현에서 생길 수 있는 두 가지 문제를 수정했습니다.

1. **Correction label 수정**  
   초기 단순 구현에서는 target correction을 `-drift`로 둘 수 있지만, rotation과 translation이 함께 있을 때 이는 정확한 SE(3) inverse가 아닙니다. 현재 코드는 `inverse_vec6(drift)`를 사용해 synthetic drift transform의 정확한 inverse correction을 label로 만듭니다.

2. **Reprojection metric 수정**  
   drift 후 image 밖으로 나간 point 때문에 단순히 projection array의 앞 N개를 비교하면 서로 다른 LiDAR point를 비교할 수 있습니다. 현재 metric은 동일한 원본 LiDAR point index가 두 projection에서 모두 image 안에 있을 때만 pixel error를 계산합니다.

Confidence head는 현재 synthetic drift magnitude에서 만든 supervised target입니다. 즉, 실제 차량의 물리적 센서 health label은 아니며, portfolio v1에서는 calibration-risk proxy로 사용하는 것이 맞습니다.
