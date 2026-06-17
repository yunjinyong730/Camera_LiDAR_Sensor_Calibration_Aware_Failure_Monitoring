# 🛣️ CalibGuard-TF-Pro

> **Camera–LiDAR Calibration Drift Detection & Recovery Monitor**  
Camera–LiDAR Calibration Drift Detection & Recovery Monitor
카메라–라이다 extrinsic calibration이 틀어졌을 때 LiDAR projection이 어떻게 무너지는지 확인하고, TensorFlow 기반 경량 모델로 6DoF 보정값을 예측해 복구 여부를 시각화하는 프로젝트입니다.

---

## Demo

### Recovery Report

<img width="2048" height="412" alt="recovery_report_model_edge_refined" src="https://github.com/user-attachments/assets/a607b93e-fd4c-4166-8318-8682405bc791" />


### Tri-color Overlay

<img width="1242" height="375" alt="tricolor_overlay_model_edge_refined" src="https://github.com/user-attachments/assets/2c45f1b4-1c85-4519-9b09-50f6c3f3e63a" />


### Demo Video



https://github.com/user-attachments/assets/ee162e54-e765-4c83-b7e8-3b2181c32e22



---

### Overview

CalibGuard-TF-Pro는 camera–LiDAR extrinsic calibration drift를 다루는 프로젝트입니다.

실제 자율주행/로보틱스 환경에서는 센서 장착 오차, 진동, 온도 변화 등으로 카메라와 라이다의 정렬이 틀어질 수 있습니다. 이 경우 LiDAR point가 이미지의 잘못된 위치에 projection되어 sensor fusion 성능이 떨어질 수 있습니다.

이 프로젝트에서는 KITTI-style 데이터를 기반으로 calibration drift를 합성하고, 모델이 이를 다시 보정할 수 있는지 확인합니다.

Normal Calibration
→ Synthetic Calibration Drift
→ Drifted Projection
→ MSF-CalibNet Correction Prediction
→ Recovered Projection
→ Recovery Visualization
Key Idea

단순히 LiDAR point를 이미지에 올리는 projection demo가 아니라, calibration이 틀어졌을 때 projection이 얼마나 무너지는지, 그리고 복구 후 normal projection에 얼마나 가까워졌는지를 확인하는 것이 목적입니다.

### Model

MSF-CalibNet은 TensorFlow 기반 경량 calibration correction model입니다.

입력은 한 프레임에서 만든 6채널 feature map입니다.

Input	Description
RGB image	카메라 이미지
Sparse depth	drifted calibration으로 projection한 LiDAR depth
Edge map	이미지 경계 정보
Residual map	LiDAR projection과 image edge의 alignment 차이

모델 출력은 다음과 같습니다.

Correction: [roll, pitch, yaw, tx, ty, tz]
Confidence: calibration confidence score

### Visualization

Tri-color overlay는 normal, drifted, recovered projection을 한 이미지에 겹쳐서 보여줍니다.

Color	Meaning
Green	정상 calibration projection
Red	drift가 들어간 projection
Cyan	보정 후 recovered projection

복구가 잘 되었다면 Cyan point가 Red point보다 Green point에 가까워져야 합니다.

### Recovery Metric

복구 여부는 reprojection error 기반으로 계산합니다.

recovery_rate = (drifted_error - recovered_error) / drifted_error * 100
Status	Rule
RECOVERED	recovery_rate >= 30%
PARTIALLY_RECOVERED	recovery_rate >= 5%
NOT_RECOVERED	recovery_rate < 5%

Toy KITTI 기반 짧은 학습에서는 NOT_RECOVERED가 나올 수 있습니다. 이는 pipeline 문제가 아니라, toy data만으로는 모델이 충분한 calibration correction을 학습하기 어렵기 때문입니다.

### Project Structure
CalibGuard-TF-Pro/
  calibguard/
    data/          # KITTI loader, drift dataset
    geometry/      # SE(3), projection, reprojection error
    features/      # RGB/depth/edge/residual feature
    models/        # TensorFlow MSF-CalibNet
    recovery/      # edge-based refinement
    metrics/       # recovery metric
    utils/         # visualization, TFLite

  scripts/
    00_make_toy_kitti.py
    01_check_projection.py
    02_train.py
    03_evaluate.py
    04_demo_recovery.py
    05_export_tflite.py
    06_streamlit_dashboard.py
    07_make_demo_video.py

### Summary

CalibGuard-TF-Pro는 camera–LiDAR extrinsic drift를 합성하고, TensorFlow 기반 경량 모델로 6DoF calibration correction을 예측한 뒤, recovered projection이 normal projection에 가까워졌는지 시각화와 reprojection error로 확인하는 프로젝트입니다.

이 프로젝트를 통해 다음을 구현했습니다.

Camera–LiDAR projection
Synthetic calibration drift generation
6DoF calibration correction prediction
Normal / Drifted / Recovered 비교
Tri-color overlay 기반 recovery visualization
Recovery rate / status 계산
Demo video 자동 생성

### Limitations
Toy KITTI는 pipeline 검증용입니다.
실제 성능 평가는 KITTI Object Detection 데이터로 진행해야 합니다.
현재 모델은 single-frame 기반입니다.
이후 temporal calibration drift detection으로 확장할 수 있습니다.
