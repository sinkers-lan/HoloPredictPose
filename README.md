# HoloPredictPose

本项目利用深度学习技术，通过模型[YOLO-X](https://github.com/Megvii-BaseDetection/YOLOX)，[Lite-HRNet](https://github.com/HRNet/Lite-HRNet)，[Videopose3D](https://github.com/facebookresearch/VideoPose3D)实时检测人体3D姿态，并基于此通过模型[TrajectoryCNN](https://github.com/lily2lab/TrajectoryCNN)预测未来人体动作。采用mmpose框架与多进程技术实现后端快速预测，利用混合现实Hololens2头戴显示器显示人物动作，做到实时抓取，实时预测，实时显示。

### 部署方式

见前端部署和后端部署

### 一些实现细节指路☞

1. 图像是如何从Hololens2（简称HL2）传递至后端的？

   使用dll获取Hololens2的传感器的图像。dll改编自[cgsaxner/HoloLens2-Unity-ResearchModeStreamer (github.com)](https://github.com/cgsaxner/HoloLens2-Unity-ResearchModeStreamer)。该项目将HoloLens2中的前景彩色图像和深度图像传输到PC端，并通过python的脚本进行接收。而我们的项目**将深度图像去除，仅使用了RGB图像**，并借鉴了[Hololens2初入——解决HL真机到PC图像传输的实时性问题_pc到hololens 屏幕镜像-CSDN博客](https://blog.csdn.net/scy261983626/article/details/116381193)的思路以改进实时性。

2. 如何使用3D人体关键点坐标控制人物，做出动作？

   [HoloPredictPose/前端源码/player_scripts.cs](https://github.com/sinkers-lan/HoloPredictPose/blob/main/前端源码/player_scripts.cs)

   该脚本提供了一个思路。使用数学的方式，将每根骨骼做两次旋转。

   我们需要进行两次旋转，这是为什么？想想一下，你自己就是人物建模。现在的需要向你左侧45度的方向鞠躬90度。那你需要①先以z轴为轴，向你的左侧左旋转45度。②再以z轴为腰部的原始位置，以z轴和你准备弯腰的方向的法向量为轴，弯腰90度。这就是两次旋转的含义。注意如果缺少了第一次旋转，直接弯腰，那你的盆骨还是超前的（这种动作大概会闪到腰吧），这是不对的。

   在运行最开始，获取每根骨骼的四元数，记为`org_q`（四元数的本质是角度，是不同于欧拉角的另一种计算和存储角度的方式）。然后计算出两次旋转的四元数，记为q1,q2。那么只需要用`q2 * q1 * org_q`即可得到骨骼旋转后的四元数。将其赋值给骨骼即可。四元数的具体原理请自行学习，我也只是现学现卖。

   假设某动作的脊椎骨的两个端点分别为A(x1,y1,z1)、B(x2,y2,z2)。设该骨骼的“原位”为垂直状态$\vec{v1}=(0,0,1)$ （不同初始方向的骨骼使用不同的单位向量即可），向量$\vec{v2}=\vec{AB}$。

   先说第二次旋转：首先算出旋转轴（使用叉乘方法`Vector3 v3 = Vector3.Cross(v1,v2)`），再计算出v1和v2的夹角（取锐角）。用四元数运算（`Quaternion q1 = Quaternion.AngleAxis((float)angle, v3)`）得到旋转四元数q1。

   第一次旋转：注意，第一次旋转，也就是弯腰的朝向，这个方向仅靠脊柱两端的A、B两点的坐标是得不到的，而需要借助盆骨两端的C、D两点计算。对于不同的骨骼需要具体问题具体分析。例如脖子的方向可以由肩膀决定，盆骨的方向却由腰决定，小腿和大腿的方向则由它们共同决定。事实上我对于这一部分做了非常精细且琐碎的分类讨论，我想这是由于3D坐标本身信息的局限性造成的，3D坐标只提供了关键点信息而非骨骼本身的信息，从而造成了这种局限性。因此如果想由3D坐标恢复出人体姿态本就有些迂回了。

