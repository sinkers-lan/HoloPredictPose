## Quick start

### Dependencies

- Python 3+

3D姿态检测依赖：

- CUDA version 10.2

- python libraries: pytorch + mmpose + mmdetection

  参考：[Installation — MMPose 1.0.0 documentation](https://mmpose.readthedocs.io/en/latest/installation.html)
  
  ![image-20230428120350326](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230428120350326.png)

动作预测模型TrajectoryCNN依赖：

- python libraries: tensorflow (>=1.0, <2.0) + opencv + numpy

网络传输部分依赖：

- python libraries: fastapi + uvicorn[standard] + python-multipart

----

### Get started

#### 后端模型测试

运行后端程序`vis_demo.py`，如果运行成功则说明模型配置成功。

----

#### 前后端连接测试

在前端Unity环境已搭建完毕的情况下，可以在部署到Hololens2前先使用`send_test.py`测试网络通信。

在前端Receive.cs脚本中，将目标host更改为后端部署环境的内网IP。8001端口为后端fastapi启动时设置的端口，可以自行更改。帧率可以自由调整。

![image-20230428125247270](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230428125247270.png)

针对两个人物unitychan/rig/offset右侧信息栏，取消勾选“Use Local Json”选项。

![image-20230428152127607](https://sinkers-pic.oss-cn-beijing.aliyuncs.com/img/image-20230428152127607.png)

设置完毕后，先运行后端`send_test.py`文件，然后在前端Unity中点击运行。观察到小人正常运动则说明通信成功。如果后端代码报错请检查后端网路传输部分环境，如果前端控制台报网络错误请检查眼镜与后端设备是否处于同一内网环境下，IP和端口填写是否正确。

----

#### 部署后运行

在Hololens2程序部署成功之后进行。有单进程方案与多进程方案，单进程方案有已知的性能问题。

##### 多进程运行

需要后端设备有至少2个GPU，因为我们在代码`try(3)`中将Lite-HRNet指定到了GPU1上运行。

将`app.py`中改为try(3)：

```python
uvicorn.run(app="try(3):app", host="0.0.0.0", port=8001, reload=True, ws="websockets", log_level="trace")
```

并将`try(3).py`中的这一行的HOST改为hololens2眼镜的内网IP，端口号不用更改：

```python
HOST = '10.21.182.65'
```

更改完毕后，先运行`try(3).py`。

当运行输出`start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`之后，将try(3)挂入后台运行，运行`app.py`。推荐使用pycharm远程连接运行代码，可以同时开启两个窗口观察控制台输出。

如果try(3)控制台输出：

```
INFO: Socket connected to ... on port ...
...
socket process put a frame.
...
socket process put a frame.
...
```

则说明运行成功。

如果仅输出第一行则建议重启Hololens2眼镜。如果都没有输出，则说明前端配置有问题，请查看前端配置文档，注意所有需要开启的权限是否开启。

前端设备请保持稳定的对准一个人物模特，模特距离眼镜至少6米。约几秒钟后，眼镜中的人物模型将会做出模特做出的动作。



##### 单进程运行（不推荐）

将`app.py`中改为try(2)：

```python
uvicorn.run(app="try(2):app", host="0.0.0.0", port=8001, reload=True, ws="websockets", log_level="trace")
```

并将`try(2).py`中的这一行的HOST改为hololens2眼镜的内网IP，端口号不用更改：

```python
HOST = '10.21.182.65'
```

然后运行`app.py`。

此时观察程序输出，如果成功则输出：

```
INFO: Socket connected to ... on port ...
```

如果输出以下内容，则说明前端配置有问题，请查看前端配置文档，注意所有需要开启的权限是否开启：

```
No frame get
```

前端设备请保持稳定的对准一个人物模特，模特距离眼镜至少6米。约几秒钟后，眼镜中的人物模型将会做出模特做出的动作。

