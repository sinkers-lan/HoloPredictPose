# -*- coding:utf-8 -*-
# 导入可视化工具包 matplotlib
import json

import typing
import matplotlib.pyplot as plt
import copy
import threading
import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import torch

from TrajectoryCNN.TrajectoryCNN_test import Model, TraCNN_predict

import cv2
import mmcv
import numpy as np
import time

from mmpose.apis import (extract_pose_sequence,
                         get_track_id, inference_pose_lifter_model,
                         inference_top_down_pose_model, init_pose_model, vis_pose_result,
                         process_mmdet_results, vis_3d_pose_result)

from mmdet.apis import inference_detector, init_detector
from videopose3D.videoPose3D_test import pose_lift

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

import asyncio

# -----------------------------网络部分定义------------------------------
import socket
import struct
import abc
import threading
# from datetime import datetime, timedelta, time
from collections import namedtuple, deque
from enum import Enum
import numpy as np
import cv2
import multiprocessing
from multiprocessing import Process, Queue
import multiprocessing as mp


def my_socket():
    # Definitions
    # Protocol Header Format
    # see https://docs.python.org/2/library/struct.html#format-characters
    VIDEO_STREAM_HEADER_FORMAT = "@qIIII18f"

    VIDEO_FRAME_STREAM_HEADER = namedtuple(
        'SensorFrameStreamHeader',
        'Timestamp ImageWidth ImageHeight PixelStride RowStride fx fy '
        'PVtoWorldtransformM11 PVtoWorldtransformM12 PVtoWorldtransformM13 PVtoWorldtransformM14 '
        'PVtoWorldtransformM21 PVtoWorldtransformM22 PVtoWorldtransformM23 PVtoWorldtransformM24 '
        'PVtoWorldtransformM31 PVtoWorldtransformM32 PVtoWorldtransformM33 PVtoWorldtransformM34 '
        'PVtoWorldtransformM41 PVtoWorldtransformM42 PVtoWorldtransformM43 PVtoWorldtransformM44 '
    )

    RM_STREAM_HEADER_FORMAT = "@qIIII16f"

    RM_FRAME_STREAM_HEADER = namedtuple(
        'SensorFrameStreamHeader',
        'Timestamp ImageWidth ImageHeight PixelStride RowStride '
        'rig2worldTransformM11 rig2worldTransformM12 rig2worldTransformM13 rig2worldTransformM14 '
        'rig2worldTransformM21 rig2worldTransformM22 rig2worldTransformM23 rig2worldTransformM24 '
        'rig2worldTransformM31 rig2worldTransformM32 rig2worldTransformM33 rig2worldTransformM34 '
        'rig2worldTransformM41 rig2worldTransformM42 rig2worldTransformM43 rig2worldTransformM44 '
    )

    # Each port corresponds to a single stream type
    VIDEO_STREAM_PORT = 23940
    AHAT_STREAM_PORT = 23941

    HOST = '10.21.182.65'

    HundredsOfNsToMilliseconds = 1e-4
    MillisecondsToSeconds = 1e-3

    class SensorType(Enum):
        VIDEO = 1
        AHAT = 2
        LONG_THROW_DEPTH = 3
        LF_VLC = 4
        RF_VLC = 5

    class FrameReceiverThread:
        def __init__(self, host, port, header_format, header_data):
            super(FrameReceiverThread, self).__init__()
            self.header_size = struct.calcsize(header_format)
            self.header_format = header_format
            self.header_data = header_data
            self.host = host
            self.port = port
            self.latest_frame = None
            self.latest_header = None
            self.socket = None
            """开始连接"""
            self.start_socket()
            """开始循环接收"""
            self.start_listen()

        def get_data_from_socket(self):
            # read the header in chunks
            reply = self.recvall(self.header_size)

            if not reply:
                print('ERROR: Failed to receive data from stream.')
                return

            data = struct.unpack(self.header_format, reply)
            header = self.header_data(*data)

            # read the image in chunks
            image_size_bytes = header.ImageHeight * header.RowStride
            image_data = self.recvall(image_size_bytes)

            return header, image_data

        def recvall(self, size):
            msg = bytes()
            while len(msg) < size:
                part = self.socket.recv(size - len(msg))
                if part == '':
                    break  # the connection is closed
                msg += part
            return msg

        def start_socket(self):
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            # send_message(self.socket, b'socket connected at ')
            print('INFO: Socket connected to ' + self.host + ' on port ' + str(self.port))

        def start_listen(self):
            t = threading.Thread(target=self.listen)
            t.start()

        """
        自定义异步接收函数
        """

        def listen_once(self):
            self.latest_header, image_data = self.get_data_from_socket()
            self.latest_frame = np.frombuffer(image_data, dtype=np.uint8).reshape((self.latest_header.ImageHeight,
                                                                                   self.latest_header.ImageWidth,
                                                                                   self.latest_header.PixelStride))
            return self.latest_frame

        """
        自定义获取最新帧
        """

        def get_last_frame(self):
            if self.latest_frame is not None:
                return self.latest_frame
            else:
                return None

        @abc.abstractmethod
        def listen(self):
            return

        @abc.abstractmethod
        def get_mat_from_header(self, header):
            return

    class VideoReceiverThread(FrameReceiverThread):
        def __init__(self, host):
            super().__init__(host, VIDEO_STREAM_PORT, VIDEO_STREAM_HEADER_FORMAT,
                             VIDEO_FRAME_STREAM_HEADER)

        def listen(self):
            print("new thread for listen")
            while True:
                self.latest_header, image_data = self.get_data_from_socket()
                self.latest_frame = np.frombuffer(image_data, dtype=np.uint8).reshape((self.latest_header.ImageHeight,
                                                                                       self.latest_header.ImageWidth,
                                                                                       self.latest_header.PixelStride))

        def get_mat_from_header(self, header):
            pv_to_world_transform = np.array(header[7:24]).reshape((4, 4)).T
            return pv_to_world_transform

    video_receiver = VideoReceiverThread(HOST)
    pass


np.warnings.filterwarnings('ignore')

app = FastAPI()


# coco->h36m转换函数
def convert_keypoint_definition(keypoints, pose_det_dataset,
                                pose_lift_dataset):
    """Convert pose det dataset keypoints definition to pose lifter dataset
    keypoints definition, so that they are compatible with the definitions
    required for 3D pose lifting.

    Args:
        keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.
        pose_det_dataset, (str): Name of the dataset for 2D pose detector.
        pose_lift_dataset (str): Name of the dataset for pose lifter model.

    Returns:
        ndarray[K, 2 or 3]: the transformed 2D keypoints.
    """
    assert pose_lift_dataset in [
        'Body3DH36MDataset', 'Body3DMpiInf3dhpDataset'
    ], '`pose_lift_dataset` should be `Body3DH36MDataset` ' \
       f'or `Body3DMpiInf3dhpDataset`, but got {pose_lift_dataset}.'

    coco_style_datasets = [
        'TopDownCocoDataset', 'TopDownPoseTrack18Dataset',
        'TopDownPoseTrack18VideoDataset'
    ]
    keypoints_new = np.zeros((17, keypoints.shape[1]), dtype=keypoints.dtype)
    if pose_lift_dataset == 'Body3DH36MDataset':
        if pose_det_dataset in ['TopDownH36MDataset']:
            keypoints_new = keypoints
        elif pose_det_dataset in coco_style_datasets:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # in COCO, head is in the middle of l_eye and r_eye
            # in PoseTrack18, head is in the middle of head_bottom and head_top
            keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
            # rearrange other keypoints
            keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        elif pose_det_dataset in ['TopDownAicDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[0] = (keypoints[9] + keypoints[6]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[8] = (keypoints[3] + keypoints[0]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # neck base (top end of neck) is 1/4 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

            keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[6, 7, 8, 9, 10, 11, 3, 4, 5, 0, 1, 2]]
        elif pose_det_dataset in ['TopDownCrowdPoseDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[0] = (keypoints[6] + keypoints[7]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[8] = (keypoints[0] + keypoints[1]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # neck base (top end of neck) is 1/4 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

            keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[7, 9, 11, 6, 8, 10, 0, 2, 4, 1, 3, 5]]
        else:
            raise NotImplementedError(
                f'unsupported conversion between {pose_lift_dataset} and '
                f'{pose_det_dataset}')

    elif pose_lift_dataset == 'Body3DMpiInf3dhpDataset':
        if pose_det_dataset in coco_style_datasets:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[11] + keypoints[12]) / 2
            # neck (bottom end of neck) is in the middle of
            # l_shoulder and r_shoulder
            keypoints_new[1] = (keypoints[5] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2

            # in COCO, head is in the middle of l_eye and r_eye
            # in PoseTrack18, head is in the middle of head_bottom and head_top
            keypoints_new[16] = (keypoints[1] + keypoints[2]) / 2

            if 'PoseTrack18' in pose_det_dataset:
                keypoints_new[0] = keypoints[1]
                # don't extrapolate the head top confidence score
                keypoints_new[16, 2] = keypoints_new[0, 2]
            else:
                # head top is extrapolated from neck and head
                keypoints_new[0] = (4 * keypoints_new[16] -
                                    keypoints_new[1]) / 3
                # don't extrapolate the head top confidence score
                keypoints_new[0, 2] = keypoints_new[16, 2]
            # arms and legs
            keypoints_new[2:14] = keypoints[[
                6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15
            ]]
        elif pose_det_dataset in ['TopDownAicDataset']:
            # head top is head top
            keypoints_new[0] = keypoints[12]
            # neck (bottom end of neck) is neck
            keypoints_new[1] = keypoints[13]
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[9] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
            # arms and legs
            keypoints_new[2:14] = keypoints[0:12]
        elif pose_det_dataset in ['TopDownCrowdPoseDataset']:
            # head top is top_head
            keypoints_new[0] = keypoints[12]
            # neck (bottom end of neck) is in the middle of
            # l_shoulder and r_shoulder
            keypoints_new[1] = (keypoints[0] + keypoints[1]) / 2
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[7] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
            # arms and legs
            keypoints_new[2:14] = keypoints[[
                1, 3, 5, 0, 2, 4, 7, 9, 11, 6, 8, 10
            ]]

        else:
            raise NotImplementedError(
                f'unsupported conversion between {pose_lift_dataset} and '
                f'{pose_det_dataset}')

    return keypoints_new


# 定义可视化图像函数，输入图像 array，可视化图像
def show_img_from_array(img):
    '''输入 array，matplotlib 可视化格式为 RGB，因此需将 BGR 转 RGB，最后可视化出来'''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


def start_multi_process():
    video_pose_3d_result = []
    trajectory_cnn_result = []

    q_frame = Queue(1)
    q_yolo_lite = Queue(1)
    q_lite_3d = Queue(1)
    q_3d_tcnn = Queue(1)

    q_2d_result = Queue(1)
    q_3d_result = Queue(1)
    q_tcnn_result = Queue(1)

    p = Process(target=my_yolo, args=(q_frame, q_yolo_lite), daemon=True, name="yolo")
    p.start()
    p = Process(target=my_lite_hr_net, args=(q_yolo_lite, q_lite_3d, q_2d_result), daemon=True, name="lite")
    p.start()
    p = Process(target=my_video_pose_3d, args=(q_lite_3d, q_3d_tcnn, q_3d_result), daemon=True, name="3d")
    p.start()
    p = Process(target=my_trajectory_cnn, args=(q_3d_tcnn, q_tcnn_result), daemon=True, name="tcnn")
    p.start()

    # 测试程序
    video_path = 'test.mp4'
    cap = cv2.VideoCapture(video_path)  # import video files
    # determine whether to open normally
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    try:
        time0 = time.time()
        while ret:
            ret, frame = cap.read()
            q_frame.put(frame, block=True, timeout=30)
            print("main process put a frame.")
            print(f"--------------------------all time:{time.time() - time0}----------------------------------")
            time0 = time.time()
            # if q_2d_result.get() is not None:
            #     video_pose_3d_result.append(q_3d_result.get())
            #     trajectory_cnn_result.append(q_tcnn_result.get())
            #     print(f"--------------------------all time:{time.time() - time0}----------------------------------")
            #     time0 = time.time()
            # else:
            #     print("---------------------------------------------------------------------------------------------")
            #     time0 = time.time()
    finally:
        cap.release()
    pose_3D_result = np.array(video_pose_3d_result)
    predict_list = np.array(trajectory_cnn_result)
    np.save("outputs/videopose3D1.npy", pose_3D_result)
    np.save("outputs/TrajectoryCNN1.npy", predict_list)


def my_yolo(q_frame: Queue, q: Queue):
    # 目标检测模型定义与初始化(可替换成yolox或者yolov7)
    det_config = 'yolox/yolox_tiny_8x8_300e_coco.py'
    det_checkpoint = 'yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
    det_model = init_detector(det_config, det_checkpoint)

    # -----------------------------进行检测
    # 检测框检测
    while True:
        frame = q_frame.get()
        print("yolo process get a frame.")
        start_time = time.time()
        mmdet_results = inference_detector(det_model, frame)
        # 提取类别 ID 为 1 的 行人 目标检测框(yolo可替换成相应的类别id)
        person_results = process_mmdet_results(mmdet_results, cat_id=1)
        end1 = time.time() - start_time
        print(f'det_test time: {end1}')

        q.put((frame, person_results))
        # print("yolo process put an result.")


def my_lite_hr_net(q_pre: Queue, q_next: Queue, q_result: Queue):
    # 人体姿态估计模型定义(lite-hrnet)
    pose_config = 'lite-hrnet/litehrnet_18_coco_256x192.py'
    pose_checkpoint = 'lite-hrnet/litehrnet18_coco_256x192-6bace359_20211230.pth'
    pose_model = init_pose_model(pose_config, pose_checkpoint)

    pose_lift_dataset = 'Body3DH36MDataset'  # videopose3D关键点标注格式
    pose_det_dataset = 'TopDownCocoDataset'  # lite-hrnet关键点标注格式

    pose_2d_results = []  # 处理后产生的2D关键点结果列表，用于后续检测(lite -> videopose3d)
    img_2d_num = 0  # 用于记录已经输出的2D关键点的个数(vediopose的局部变量)
    pose_2D = []  # 存储2D关键点的所有结果(lite最终结果)

    pose_results = []
    next_id = 0  # 用于标注tracked_id
    pose_results_last = pose_results  # 用于添加track_id

    while True:
        frame, person_results = q_pre.get()
        # print("lite process get an result.")
        # 关键点检测
        start_time = time.time()
        pose_results, returned_outputs = inference_top_down_pose_model(pose_model, frame, person_results,
                                                                       bbox_thr=0.3,
                                                                       format='xyxy', dataset='TopDownCocoDataset')
        # 过滤的工作是否可以换到yolo进程中完成
        end1 = time.time() - start_time
        print(f'lite-hrnet_test time: {end1}')
        pose_2D.append(copy.deepcopy(pose_results[0]['keypoints'][:, :2]))
        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id)

        img_2d_num += 1  # 结果数目加一
        if img_2d_num <= 10:
            pose_2d_results.append(copy.deepcopy(pose_results))  # 添加到结果列表中，用于后续预测
        else:
            pose_2d_results.pop(0)
            pose_2d_results.append(copy.deepcopy(pose_results))

        if img_2d_num >= 10:  # 已经积攒够了所需的帧数
            if img_2d_num == 10:
                # convert keypoint definition
                for pose_det_results in pose_2d_results:
                    for res in pose_det_results:
                        keypoints = res['keypoints']
                        res['keypoints'] = convert_keypoint_definition(
                            keypoints, pose_det_dataset, pose_lift_dataset)
            else:  # img_2d_num > 10
                # convert keypoint definition
                for res in pose_2d_results[-1]:
                    keypoints = res['keypoints']
                    res['keypoints'] = convert_keypoint_definition(
                        keypoints, pose_det_dataset, pose_lift_dataset)
            q_next.put((frame, pose_2d_results))
            # q_result.put(pose_2d_results)
        else:
            # q_result.put(None)
            pass


def my_video_pose_3d(q_pre: Queue, q_next: Queue, q_result: Queue):
    # videopose3D模型定义
    pose_lifter_config = "videopose3D/videopose3d_h36m_27frames_fullconv_semi-supervised_cpn_ft.py"
    pose_lifter_checkpoint = "videopose3D/videopose_h36m_27frames_fullconv_semi-supervised_cpn_ft-71be9cde_20210527.pth"
    pose_lift_model = init_pose_model(pose_lifter_config, pose_lifter_checkpoint)

    pose_2d_results = []  # 处理后产生的2D关键点结果列表，用于后续检测(lite -> videopose3d)
    img_2d_num = 0  # 用于记录已经输出的2D关键点的个数(vediopose的局部变量)
    pose_3D_result = []

    # ----------------------------------第二阶段:人体3D关键点估计与动作预测-------------------------------------
    while True:
        frame, pose_2d_results = q_pre.get()
        # videoPose3D预测
        start_time = time.time()
        img_size = (frame.shape[1], frame.shape[0])
        pose_3D = pose_lift(pose_lift_model, pose_2d_results, img_size)
        end1 = time.time() - start_time
        print(f'videopose3D_test time: {end1}')
        # pose_3D_result.append(pose_3D[0].copy())
        # q_result.put(pose_3D[0].copy())

        # 调整videoPose3D的输出使其满足TrajectoryCNN的输入
        pose_3D_ndarry = np.ndarray((len(pose_3D), 17, 3))
        for i, pose in enumerate(pose_3D):
            pose = pose[:, :3]
            pose_3D_ndarry[i] = copy.deepcopy(pose)
        pose_3D_ndarry *= 1000
        q_next.put(pose_3D_ndarry)


def my_trajectory_cnn(q: Queue, q_result: Queue):
    # TrajectoryCNN模型定义
    pose3D_predict_model = tcnn.Model()

    # 最终结果
    predict_list = []  # 存储最后的预测结果序列(tcnn最终结果)

    while True:
        pose_3D_ndarry = q.get()
        # TraCNN动作预测
        predict_3D = tcnn.TraCNN_predict(pose3D_predict_model, pose_3D_ndarry)
        # predict_list.append(predict_3D.copy())  # 积攒结果
        # q_result.put(predict_3D.copy())


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/cv")
async def get_cv():
    frame = video_receiver.latest_frame
    if frame is None:
        print("No frame get")
        return {"org": None, "pred": None, "avail": 0}
    return models(frame)


@app.get("/start")
async def get_start():
    # manager.start()
    pass


@app.get("/end")
async def get_end():
    # manager.end()
    pass

    # 日志级别log_level="trace"
