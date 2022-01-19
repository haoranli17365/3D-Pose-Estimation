# 3DPoseEstimation


### Requirements 
```
pip install opencv-python matplotlib
```

### Download Models
See Release https://github.com/Ascend-Huawei/3DPoseEstimation/releases/tag/v0

Image to 2D Keypoints Model
```
# Under root of this repo
wget -nc --no-check-certificate 'https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/body_pose_picture/OpenPose_light.pb' -O model/OpenPose_light.pb

# Convert
atc --framework=3 --model=model/OpenPose_light.pb --input_shape="input001:1,368,368,3" --input_format=NHWC --output=model/OpenPose_light --output_type=FP32 --out_nodes="light_openpose/stage_1/ArgMax:0" --soc_version=Ascend310

```

2D to 3D Lifting Model (Pose3D RIE)
``` 
# See Release
wget -nc --no-check-certificate https://github.com/Ascend-Huawei/3DPoseEstimation/releases/download/v0/pose3d_rie_sim.onnx -O model/pose3d_rie_sim.onnx

# Convert
atc --input_shape="0:1,243,17,2" --input_format=NCHW --output="model/pose3d_rie_sim" --soc_version=Ascend310 --framework=5 --model="model/pose3d_rie_sim.onnx"
```

### Sample Run
```
python run.py \
    --model2D='./model/OpenPose_light.om' \
    --model3D ./model/pose3d_rie_sim.om \
    --input ./data/pose3d_test_10s.mp4 \
    --output_dir='./outputs'
```
