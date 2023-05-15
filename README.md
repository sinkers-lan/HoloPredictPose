# HoloPredictPose

本项目利用深度学习技术，通过模型[YOLO-X](https://github.com/Megvii-BaseDetection/YOLOX)，[Lite-HRNet](https://github.com/HRNet/Lite-HRNet)，[Videopose3D](https://github.com/facebookresearch/VideoPose3D)实时检测人体3D姿态，并基于此通过模型[TrajectoryCNN](https://github.com/lily2lab/TrajectoryCNN)预测未来人体动作。采用mmpose框架与多进程技术实现后端快速预测，利用混合现实Hololens2头戴显示器显示人物动作，做到实时抓取，实时预测，实时显示。

### 部署方式

见[前端部署](https://github.com/sinkers-lan/HoloPredictPose/blob/main/%E5%89%8D%E7%AB%AF%E6%BA%90%E7%A0%81/%E5%89%8D%E7%AB%AF%E9%83%A8%E7%BD%B2.md)和[后端部署](https://github.com/sinkers-lan/HoloPredictPose/blob/main/%E5%90%8E%E7%AB%AF%E6%BA%90%E7%A0%81/%E5%90%8E%E7%AB%AF%E9%83%A8%E7%BD%B2.md)

