# HoloPredictPose

前端环境部署教程

[toc]



## 前言

在这个项目中，我们运用Hololens2的传感器实时获取人体图像数据，传输给后端进行人体姿态预测模型的处理。同时，我们基于unity进行开发，将unity部署到Hololens2上，以便于观察人体真实动作与模型预测动作的实时效果。
本文的配置将实现把Hololens2设备上的图像数据通过wifi实时地传递到电脑端，方便用深度学习算法对图像进行处理，同时，实现unity动作效果的实时展示。



## 安装工具

1. Windows10 或 Windows11操作系统

2. 一台Hololens2头戴显示设备

3. WIN SDK（最新版）

4. UnityHub & Unity 2020.3.45  [[下载地址](https://unity.cn/releases/full/2020)]

5. Visual Studio 2022 [[下载地址](https://visualstudio.microsoft.com/zh-hans/)]

   **必备插件：**

   ![1](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/1.png)
   
   单个组件：USB设备连接性
   
   ![image-20230428180449321](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230428180449321.png)



## 前端项目使用说明

### Unity环境配置与资源导入

①官网下载unity hub

②在hub内下载unity2020.3版本，需要勾选以下模块：

![image-20230427125427514](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230427125427514.png)

③新建unity3Dcore项目，项目名称可以指定为HoloPredictPose

![image-20230427125705194](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230427125705194.png)

④“资源”——“导入包”——“自定义包”，导入HoloPredict.unitypackage

⑤“窗口”——“包管理器”，左上角拉选“Unity注册表”，分别搜索并下载以下三个包：Collections、Animation Rigging、Newtonsoft Json

![image-20230427130855228](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230427130855228.png)

⑥观察控制台没有编译错误，则导入成功

⑦project->Assets->Scenes下双击“0.unity”，即可看到项目预设场景

![image-20230428151654394](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230428151654394.png)



### Unity配置测试

#### 本地测试

资源导入成功后，点击上方三角按钮播放，观察人物动作变化，然后从“游戏”窗口切换到“场景”。若画面中人物动作正常变化，则项目搭建成功，可以进入下一步。

#### 联机测试

*必做步骤，同见[HolopredictPose/后端代码/后端部署.md](https://github.com/sinkers-lan/HoloPredictPose/blob/main/%E5%90%8E%E7%AB%AF%E6%BA%90%E7%A0%81/%E5%90%8E%E7%AB%AF%E9%83%A8%E7%BD%B2.md#%E5%89%8D%E5%90%8E%E7%AB%AF%E8%BF%9E%E6%8E%A5%E6%B5%8B%E8%AF%95)。在前端Unity环境已搭建完毕的情况下，应在部署到Hololens2前先使用后端代码`send_test.py`测试网络通信。

首先，层级窗口中找到![image-20230428183710295](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230428183710295.png)对象并选中，在检查器窗口找到Receive脚本，将目标host更改为后端部署环境的内网IP。8001端口为后端fastapi启动时设置的端口，可以自行更改。帧率可以自由调整。

![image-20230428125247270](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230428125247270.png)

**分别**针对两个对象unitychan和unitychan(1)，展开/rig/offsets![image-20230428183817957](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230428183817957.png)，选中offsets对象，在检查器窗口找到Player_scripts脚本，取消勾选“Use Local Json”选项。

![image-20230428152127607](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230428152127607.png)

设置完毕后，先运行后端`send_test.py`文件，然后在前端Unity中点击运行。观察到小人正常运动则说明通信成功。如果后端代码报错请检查后端网路传输部分环境，如果前端控制台报网络错误请检查眼镜与后端设备是否处于同一内网环境下，IP和端口填写是否正确。



### 生成与运行

#### 更改生成相关设置

*以下操作的目的是将配置好的项目编译并导入到Hololens2眼镜中

①导入混合现实工具包MRTK，参考教程 [[MRTK导入-参考教程](https://learn.microsoft.com/zh-cn/training/modules/learn-mrtk-tutorials/1-5-exercise-configure-resources?ns-enrollment-type=learningpath&ns-enrollment-id=learn.azure.beginner-hololens-2-tutorials&tabs=openxr)]，按照教程步骤做完`导入MRTK Unity基础包`小节的全部内容。

②在“文件”->“生成设置”（File->Build Settings）中，左侧选择”Universal Windows Platform“，点击”切换平台“。然后依次选择以下配置：

![image-20230427153012542](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230427153012542.png)

③点击”玩家设置“（File->Project Settings->Player）

3.1 更改包名，也可以使用默认包名但不建议，此步骤是为了防止相同包名的应用相互覆盖

![image-20230427153522220](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230427153522220.png)

3.2 在Capability中选择以下选项，该步骤的目的使得应用获取网络传输等权限，已经勾选的选项不用取消勾选：

![image-20230427153644030](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230427153644030.png)

![image-20230427154422842](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230427154422842.png)

④为dll库更改平台设置如下：

该dll的作用是流式获取Hololens2的Camera传感器的图像。该dll由[cgsaxner/HoloLens2-Unity-ResearchModeStreamer (github.com)](https://github.com/cgsaxner/HoloLens2-Unity-ResearchModeStreamer)提供，并进行了优化以适应本项目。

![image-20230428134422848](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230428134422848.png)

#### 生成与部署

##### 生成

“文件”——“生成设置”——“生成”，新建一个空文件夹，选择文件夹开始生成。

##### Hololens2配置

参考[使用 Visual Studio 进行部署和调试 - Mixed Reality | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/mixed-reality/develop/advanced-concepts/using-visual-studio?tabs=hl2)

必须完成的步骤包括：

- Hololens2打开设置->更新和安全->面向开发人员->开启Developer Mode->开启device protal

  ![Enabling developer mode in the Settings app for Windows Holographic](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/using-windows-portal-img-01.jpg)

  ![img](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/deviceportal_usbncm_ipaddress.jpg)

- 将上图黄框地址输入电脑浏览器，或对Hololens2使用语音命令使用语音命令“我的 IP 地址是什么？”查看Hololens2的IP地址并输入电脑浏览器，打开设备门户。首次连接到 HoloLens 上的设备门户时，需要创建用户名和密码。**开启开发者模式的传感器流访问权限**。

  ![image-20230428203231754](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230428203231754.png)

##### 生成解决方案

生成完毕后，打开生成文件夹，双击.sln项目，用visual stutio打开。

将Hololens2用usb链接至计算机，修改设置如下：

![image-20230428184613774](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230428184613774.png)

点击“生成”——“生成解决方案”/“重新生成解决方案"。

生成成功之后，点击”调试“——”开始执行（不调试）“，该操作会将解决方案部署到HoloLens2上，并启动应用。首次执行会要求配对。

以后可以从Hololens2上呼出主面板，点击应用，寻找部署好的应用并启动。

