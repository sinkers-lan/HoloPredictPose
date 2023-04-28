# -*- coding:utf-8 -*-
# 导入可视化工具包 matplotlib
import time

import matplotlib.pyplot as plt
import copy
import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import torch

from TrajectoryCNN.TrajectoryCNN_test import Model, TraCNN_predict

import cv2
import mmcv
import numpy as np

from mmpose.apis import (extract_pose_sequence,
                         get_track_id, inference_pose_lifter_model,
                         inference_top_down_pose_model, init_pose_model, vis_pose_result,
                         process_mmdet_results, vis_3d_pose_result)

from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo
from mmpose.models import PoseLifter, TopDown
from mmdet.apis import inference_detector, init_detector
from videopose3D.videoPose3D_test import pose_lift


# 定义可视化图像函数，输入图像路径，可视化图像
def show_img_from_path(img_path):
    '''opencv 读入图像，matplotlib 可视化格式为 RGB，因此需将 BGR 转 RGB，最后可视化出来'''
    img = cv2.imread(img_path)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


# 定义可视化图像函数，输入图像 array，可视化图像
def show_img_from_array(img):
    '''输入 array，matplotlib 可视化格式为 RGB，因此需将 BGR 转 RGB，最后可视化出来'''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


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


# ---------------------------测试代码:模拟远方发来的图片------------------------------
import requests
import json
import cv2
import base64


def getByte(path):
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())  # 二进制读取后变base64编码
    img_str = img_byte.decode('ascii')  # 转成python的unicode
    return img_str


# ----------------------------------相关变量定义-------------------------------------------

# ------------------------TrajectoryCNN模型相关变量定义
# test_path = 'TrajectoryCNN/data/h36m20/my_test/h36m2.npy'
# result_path = "TrajectoryCNN/results/h36m/v2"
# model_path = 'TrajectoryCNN/checkpoints/h36m/v1/model.ckpt-769500'
# input_length = 10
# seq_length = 20
# joints_number = 14
# joint_dims = 3
# stacklength = 4
# filter_size = 3
# batch_size = 1
# n_gpu = 2
# num_hidden = [64, 64, 64, 64, 64]


def main():
    # -----------------------------模型的定义与初始化
    # 目标检测模型定义与初始化(可替换成yolox或者yolov7)
    det_config = 'yolox/yolox_tiny_8x8_300e_coco.py'
    det_checkpoint = 'yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
    det_model = init_detector(det_config, det_checkpoint)

    # 人体姿态估计模型定义(lite-hrnet)
    pose_config = 'lite-hrnet/litehrnet_18_coco_256x192.py'
    pose_checkpoint = 'lite-hrnet/litehrnet18_coco_256x192-6bace359_20211230.pth'
    pose_model = init_pose_model(pose_config, pose_checkpoint)

    # videopose3D模型定义
    pose_lifter_config = "videopose3D/videopose3d_h36m_27frames_fullconv_semi-supervised_cpn_ft.py"
    pose_lifter_checkpoint = "videopose3D/videopose_h36m_27frames_fullconv_semi-supervised_cpn_ft-71be9cde_20210527.pth"
    pose_lift_model = init_pose_model(pose_lifter_config, pose_lifter_checkpoint)

    # TrajectoryCNN模型定义
    pose3D_predict_model = Model()

    # ------------------------模型间通讯变量定义
    img_path = 'data/test.jpg'  # 初始图片读取路径
    pose_2d_results = []  # 处理后产生的2D关键点结果列表，用于后续检测
    img_2d_num = 0  # 用于记录已经输出的2D关键点的个数
    # img_size = (1280, 720)  # 输入图片尺寸
    pose_det_dataset = 'TopDownCocoDataset'  # lite-hrnet关键点标注格式
    pose_lift_dataset = 'Body3DH36MDataset'  # videopose3D关键点标注格式
    # pose_3D = []  # 存储videopose3D输出的3D关键点，用于后续模型预测
    predict_list = []  # 存储最后的预测结果序列
    pose_3D_result = []  # 存储3D关键点结果
    pose_2D = []  # 存储2D关键点的所有结果

    next_id = 0  # 用于标注tracked_id
    pose_results = []

    # 可视化测试
    # 数据集格式获取，用于画图
    pose_lift_dataset_info = pose_lift_model.cfg.data['test'].get(
        'dataset_info', None)
    if pose_lift_dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        pose_lift_dataset_info = DatasetInfo(pose_lift_dataset_info)
    # 视频信息获取，用于可视化测试
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = None
    writer = None

    # 测试程序
    video_path = 'test.mp4'
    cap = cv2.VideoCapture(video_path)  # import video files

    # 帧率(frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # determine whether to open normally
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    count = 0  # count the number of pictures
    # frame_interval_count = 0
    # loop read video frame
    while ret:
        count += 1
        print("--------------------------------读取到第{}帧------------------".format(count))

        # ----------------------------------start--------------------------------
        pose_results_last = pose_results  # 用于添加track_id

        # ----------------------------------第一阶段:人体2D关键点预测--------------------------------
        img_size = (frame.shape[1], frame.shape[0])
        # -----------------------------进行检测
        # show_img_from_path(img_path)
        # 检测框检测
        start_time = time.time()
        mmdet_results = inference_detector(det_model, frame)
        # 提取类别 ID 为 1 的 行人 目标检测框(yolo可替换成相应的类别id)
        end1 =time.time() - start_time
        print(f'det_test time: {end1}')

        person_results = process_mmdet_results(mmdet_results, cat_id=1)
        print("bbox检测完成")
        # 关键点检测
        start_time = time.time()
        pose_results, returned_outputs = inference_top_down_pose_model(pose_model, frame, person_results,
                                                                       bbox_thr=0.3,
                                                                       format='xyxy', dataset='TopDownCocoDataset')
        end1 =time.time() - start_time
        print(f'lite-hrnet_test time: {end1}')
        pose_2D.append(copy.deepcopy(pose_results[0]['keypoints'][:, :2]))

        # # 可视化测试
        # vis_result = vis_pose_result(pose_model,
        #                              frame,
        #                              pose_results,
        #                              radius=8,
        #                              thickness=3,
        #                              dataset='TopDownCocoDataset',
        #                              show=False)
        # show_img_from_array(vis_result)
        # cv2.imwrite('outputs/2D_test_result.jpg', vis_result)

        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id)

        img_2d_num += 1  # 结果数目加一
        if img_2d_num <= 10:
            # list.append(obj.copy())
            pose_2d_results.append(copy.deepcopy(pose_results))  # 添加到结果列表中，用于后续预测
        else:
            pose_2d_results = pose_2d_results[1:]
            pose_2d_results.append(copy.deepcopy(pose_results))

        # ----------------------------------第二阶段:人体3D关键点估计与动作预测-------------------------------------
        if img_2d_num >= 10:  # 已经积攒够了所需的帧数
            if img_2d_num == 10:
                # convert keypoint definition
                for pose_det_results in pose_2d_results:
                    for res in pose_det_results:
                        keypoints = res['keypoints']
                        res['keypoints'] = convert_keypoint_definition(
                            keypoints, pose_det_dataset, pose_lift_dataset)
            else: #img_2d_num > 10
                # convert keypoint definition
                for res in pose_2d_results[-1]:
                    keypoints = res['keypoints']
                    res['keypoints'] = convert_keypoint_definition(
                        keypoints, pose_det_dataset, pose_lift_dataset)


            # videoPose3D预测
            start_time = time.time()
            pose_3D = pose_lift(pose_lift_model, pose_2d_results, img_size)
            end1 = time.time() - start_time
            print(f'videopose3D_test time: {end1}')
            pose_3D_result.append(pose_3D[0].copy())

            # 调整videoPose3D的输出使其满足TrajectoryCNN的输入
            pose_3D_ndarry = np.ndarray((len(pose_3D), 17, 3))
            for i, pose in enumerate(pose_3D):
                pose = pose[:, :3]
                pose_3D_ndarry[i] = copy.deepcopy(pose)
            pose_3D_ndarry *= 1000

            # for i in range(2):
            #     pose_3D_data = pose_3D_ndarry[i*10:(i+1)*10]
            # TraCNN动作预测
            predict_3D = TraCNN_predict(pose3D_predict_model, pose_3D_ndarry)

            # 可视化
            predict_list.append(predict_3D.copy())  # 积攒结果
            # ----------------------------------end--------------------------------

            # 可视化
            # Pose processing
            # 1.pose_3D
            # 2.predict_3D
            pose_vis_list = []
            for i in range(2):
                if i == 0:  # 1.pose_3D
                    res = {}
                    keypoints_3d = pose_3D[0].copy()[:, :3]
                    # exchange y,z-axis, and then reverse the direction of x,z-axis
                    keypoints_3d = keypoints_3d[..., [0, 2, 1]]
                    keypoints_3d[..., 0] = -keypoints_3d[..., 0]
                    keypoints_3d[..., 2] = -keypoints_3d[..., 2]
                    # rebase height (z-axis)
                    keypoints_3d[..., 2] -= np.min(
                        keypoints_3d[..., 2], axis=-1, keepdims=True)
                    res['keypoints_3d'] = copy.deepcopy(keypoints_3d)
                    # add title
                    res['title'] = "videopose3D_output"
                    # only visualize the target frame
                    res['keypoints'] = None
                    res['bbox'] = None
                    res['track_id'] = 0
                    pose_vis_list.append(copy.deepcopy(res))
                else:  # 2.predict_3D
                    res = {}
                    keypoints_3d = predict_3D.copy().reshape((17, 3))
                    # exchange y,z-axis, and then reverse the direction of x,z-axis
                    keypoints_3d = keypoints_3d[..., [0, 2, 1]]
                    keypoints_3d[..., 0] = -keypoints_3d[..., 0]
                    keypoints_3d[..., 2] = -keypoints_3d[..., 2]
                    # rebase height (z-axis)
                    keypoints_3d[..., 2] -= np.min(
                        keypoints_3d[..., 2], axis=-1, keepdims=True)
                    res['keypoints_3d'] = copy.deepcopy(keypoints_3d)
                    # add title
                    res['title'] = "TrajectoryCNN_output"
                    # only visualize the target frame
                    res['keypoints'] = None
                    res['bbox'] = None
                    res['track_id'] = 0
                    pose_vis_list.append(copy.deepcopy(res))

            # Visualization
            # if num_instances < 0:
            #     num_instances = len(pose_vis_list)
            num_instances = len(pose_vis_list)
            img_vis = vis_3d_pose_result(
                pose_lift_model,
                result=pose_vis_list,
                dataset=pose_lift_dataset,
                dataset_info=pose_lift_dataset_info,
                out_file=None,
                num_instances=num_instances)

            # if save_out_video:
            if writer is None:
                writer = cv2.VideoWriter("test_output_old.mp4", fourcc,
                                         fps, (img_vis.shape[1], img_vis.shape[0]))
            writer.write(img_vis)
        # 读取下一帧
        ret, frame = cap.read()

    pose_3D_result = np.array(pose_3D_result)
    predict_list = np.array(predict_list)
    pose_2D = np.array(pose_2D)
    np.save("outputs/videopose3D.npy", pose_3D_result)
    np.save("outputs/TrajectoryCNN.npy", predict_list)
    np.save("outputs/pose_2D_demo.npy", pose_2D)
    cap.release()
    # if save_out_video:
    writer.release()


if __name__ == '__main__':
    main()
