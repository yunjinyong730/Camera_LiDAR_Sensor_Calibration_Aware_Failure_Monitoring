# CalibGuard-TF-Pro

> **Camera–LiDAR Calibration Drift Detection & Recovery Monitor**  
> 카메라–라이다 extrinsic calibration이 틀어졌을 때 projection이 어떻게 무너지는지 확인하고, TensorFlow 기반 경량 모델로 6DoF 보정값을 예측해 복구 여부를 시각화하는 프로젝트입니다.

---

## Demo

### Recovery Report

(여기에 `assets/recovery_report_model_edge_refined.png` 넣기)

```markdown
![Recovery Report](assets/recovery_report_model_edge_refined.png)
```

### Tri-color Overlay

(여기에 `assets/tricolor_overlay_model_edge_refined.png` 넣기)

```markdown
![Tri-color Overlay](assets/tricolor_overlay_model_edge_refined.png)
```

### Demo Video

(여기에 `assets/calibguard_demo.mp4` 넣기)

```markdown
[Demo Video](assets/calibguard_demo.mp4)
```

---

## What is CalibGuard?

CalibGuard는 KITTI-style camera–LiDAR 데이터를 이용해 다음 과정을 수행합니다.

```text
Normal Calibration
→ Synthetic Calibration Drift
→ Drifted LiDAR Projection
→ MSF-CalibNet Correction Prediction
→ Recovered Projection
→ Recovery Metric / Visualization
```

핵심 목표는 단순 projection demo가 아니라, **센서 보정 오류가 perception/fusion에 미치는 영향을 감지하고 복구하는 것**입니다.

---

## Key Features

- KITTI-style camera–LiDAR projection
- Synthetic 6DoF calibration drift injection
- TensorFlow 기반 경량 보정 모델 `MSF-CalibNet`
- 6DoF correction 예측: `[roll, pitch, yaw, tx, ty, tz]`
- Normal / Drifted / Recovered projection 비교
- Tri-color overlay 시각화
- Error vector 기반 recovery 확인
- Recovery rate / status 자동 계산
- Demo video 자동 생성
- TFLite export 지원

---

## Model Input / Output

### Input

MSF-CalibNet은 한 프레임을 6채널 tensor로 변환해 입력받습니다.

| Channel | Description |
|---|---|
| RGB image | 카메라 이미지 |
| Sparse depth map | drifted calibration으로 projection한 LiDAR depth |
| Edge map | 이미지 경계 정보 |
| Residual map | LiDAR point와 image edge의 alignment residual |

```text
Input: H x W x 6
```

### Output

```text
Correction: [roll, pitch, yaw, tx, ty, tz]
Confidence: calibration confidence score
```

---

## How to Read the Visualization

Tri-color overlay의 색상 의미는 다음과 같습니다.

| Color | Meaning |
|---|---|
| Green | Normal projection |
| Red | Drifted projection |
| Cyan | Recovered projection |

복구가 잘 되었다면:

```text
Cyan points should move closer to Green points than Red points.
```

즉, **Cyan이 Green에 가까워질수록 calibration recovery가 잘 된 것**입니다.

---

## Recovery Metric

Recovery rate는 drifted error가 recovered error에서 얼마나 줄었는지로 계산합니다.

```text
recovery_rate = (drifted_error - recovered_error) / drifted_error * 100
```

| Status | Rule |
|---|---|
| RECOVERED | recovery_rate >= 30% |
| PARTIALLY_RECOVERED | recovery_rate >= 5% |
| NOT_RECOVERED | recovery_rate < 5% |

Toy KITTI의 짧은 학습에서는 `NOT_RECOVERED`가 나올 수 있습니다.  
이는 pipeline 문제가 아니라 모델이 아직 충분히 학습되지 않았다는 의미입니다.

---

## Project Structure

```text
CalibGuard-TF-Pro/
  calibguard/
    data/          # KITTI loader, drift dataset
    geometry/      # SE(3), projection, reprojection error
    features/      # RGB/depth/edge/residual input builder
    models/        # TensorFlow MSF-CalibNet
    recovery/      # edge-based refinement
    metrics/       # recovery metric
    utils/         # visualization, TFLite, profiler

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
    # 여기에 README에 넣을 이미지/영상 파일 넣기
```

---

## Asset Placement

GitHub README에 결과를 보이게 하려면 아래처럼 파일을 넣으면 됩니다.

```text
assets/
  recovery_report_model_edge_refined.png   # Recovery Report 이미지 넣기
  tricolor_overlay_model_edge_refined.png  # Green/Red/Cyan overlay 이미지 넣기
  calibguard_demo.mp4                      # 자동 생성된 demo video 넣기
  normal_projection.png                    # 선택: normal projection 이미지 넣기
  drifted_projection.png                   # 선택: drifted projection 이미지 넣기
  recovered_projection.png                 # 선택: recovered projection 이미지 넣기
```

---

## Quick Start: Toy KITTI

```powershell
conda env create -f environment.yml
conda activate calibguard-tf
python -m pip install -e .
```

```powershell
python scripts/00_make_toy_kitti.py --out_dir data/toy_kitti/training --num_frames 24
```

```powershell
python scripts/01_check_projection.py --data_root data/toy_kitti/training --frame_id 000000 --out_dir outputs/toy_projection_check
```

```powershell
python scripts/02_train.py --data_root data/toy_kitti/training --out_dir runs/toy_msf_calibnet --epochs 50 --batch_size 4 --steps_per_epoch 100 --val_steps 20 --img_h 128 --img_w 384
```

```powershell
python scripts/04_demo_recovery.py --data_root data/toy_kitti/training --model_path runs/toy_msf_calibnet/best.keras --frame_id 000000 --out_dir outputs/toy_recovery_demo --img_h 128 --img_w 384 --recovery_mode model_edge_refined
```

```powershell
python scripts/07_make_demo_video.py --data_root data/toy_kitti/training --model_path runs/toy_msf_calibnet/best.keras --out_dir outputs/toy_demo_video --num_frames 24 --fps 4 --img_h 128 --img_w 384 --recovery_mode model_edge_refined --ramp_drift
```

---

## Real KITTI Training

실제 KITTI Object Detection 데이터는 아래 구조로 준비합니다.

```text
data/kitti_object/training/
  image_2/
  velodyne/
  calib/
  label_2/
```

실제 학습:

```powershell
python scripts/02_train.py --data_root data/kitti_object/training --out_dir runs/kitti_msf_calibnet_real --epochs 50 --batch_size 8 --steps_per_epoch 1000 --val_steps 150 --img_h 192 --img_w 640
```

평가:

```powershell
python scripts/03_evaluate.py --data_root data/kitti_object/training --model_path runs/kitti_msf_calibnet_real/best.keras --out_csv runs/kitti_msf_calibnet_real/eval.csv --num_samples 500 --img_h 192 --img_w 640
```

데모 생성:

```powershell
python scripts/04_demo_recovery.py --data_root data/kitti_object/training --model_path runs/kitti_msf_calibnet_real/best.keras --frame_id 000000 --out_dir outputs/kitti_recovery_demo_real --img_h 192 --img_w 640 --recovery_mode model_edge_refined
```

---

## Portfolio Summary

> CalibGuard-TF-Pro는 camera–LiDAR extrinsic drift를 합성하고, TensorFlow 기반 경량 모델로 6DoF calibration correction을 예측한 뒤, recovered projection이 normal projection으로 실제 가까워졌는지 reprojection error, tri-color overlay, error vector, recovery rate로 검증하는 프로젝트입니다.

---

## Limitations

- Toy KITTI는 pipeline 검증용입니다.
- 최종 성능은 실제 KITTI 학습으로 평가해야 합니다.
- 현재 모델은 single-frame 기반입니다.
- 다음 단계는 temporal calibration encoder, KITTI Tracking, ROS2/C++ integration입니다.
