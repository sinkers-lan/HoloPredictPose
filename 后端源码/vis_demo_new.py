# -*- coding:utf-8 -*-
# 导入可视化工具包 matplotlib
import time

import matplotlib.pyplot as plt
import copy

import TrajectoryCNN.TrajectoryCNN_test as tcnn

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
from videopose3D.videoPose3D_test_2 import pose_lift
from convert_keypoint_definition import convert_keypoint_definition

from multiprocessing import Process, Queue
import multiprocessing as mp
import threading
import queue


def start_multi_process():
    mp.set_start_method('spawn', force=True)

    lite_hrnet_2d_result = []
    video_pose_3d_result = []
    trajectory_cnn_result = []

    q_frame = Queue(1)
    q_yolo_lite = Queue(1)
    q_lite_3d = Queue(1)
    q_3d_tcnn = Queue(1)

    q_2d_result = Queue()
    q_3d_result = Queue()
    q_tcnn_result = Queue()

    p0 = Process(target=my_socket, args=(q_frame,), daemon=True, name="socket")
    p0.start()
    p1 = Process(target=my_yolo, args=(q_frame, q_yolo_lite), daemon=True, name="yolo")
    p1.start()
    p2 = Process(target=my_lite_hr_net, args=(q_yolo_lite, q_lite_3d, q_2d_result), daemon=True, name="lite")
    p2.start()
    p3 = Process(target=my_video_pose_3d, args=(q_lite_3d, q_3d_tcnn, q_3d_result, q_2d_result), daemon=True, name="3d")
    p3.start()
    p4 = Process(target=my_trajectory_cnn, args=(q_3d_tcnn, q_tcnn_result), daemon=True, name="tcnn")
    p4.start()

    p0.join()

    while not q_2d_result.empty():
        lite_hrnet_2d_result.append(q_2d_result.get().reshape(17, 3))
    while not q_3d_result.empty():
        video_pose_3d_result.append(q_3d_result.get().reshape(17, 3))
        print(len(video_pose_3d_result))
    while not q_tcnn_result.empty():
        trajectory_cnn_result.append(q_tcnn_result.get().reshape(17, 3))


    pose_3D_result = np.array(video_pose_3d_result)
    predict_list = np.array(trajectory_cnn_result)
    np.save("outputs/videopose3D1.npy", pose_3D_result)
    np.save("outputs/TrajectoryCNN1.npy", predict_list)
    np.save("outputs/2d_transform.py", np.array(lite_hrnet_2d_result))
    print("done!")


def my_socket(q_frame: Queue):
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

    try:
        while ret:
            if q_frame.full():
                try:
                    q_frame.get(block=False)
                except queue.Empty:
                    pass
            print("socket process put a frame.")
            q_frame.put(frame, block=True, timeout=30)
            time.sleep(1 / fps)  # 按原帧率播放
            ret, frame = cap.read()
    finally:
        cap.release()
    pass


def my_yolo(q_frame: Queue, q: Queue):
    # 目标检测模型定义与初始化(可替换成yolox或者yolov7)
    det_config = 'yolox/yolox_tiny_8x8_300e_coco.py'
    det_checkpoint = 'yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
    det_model = init_detector(det_config, det_checkpoint)

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
        if q.full():
            try:
                q.get(block=False)
            except queue.Empty:
                pass
        q.put((frame, person_results))
        print("yolo process put a frame.")


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
        if img_2d_num <= 27:
            pose_2d_results.append(copy.deepcopy(pose_results))  # 添加到结果列表中，用于后续预测
        else:
            pose_2d_results.pop(0)
            pose_2d_results.append(copy.deepcopy(pose_results))

        for res in pose_2d_results[-1]:
            keypoints = res['keypoints']
            res['keypoints'] = convert_keypoint_definition(
                keypoints, pose_det_dataset, pose_lift_dataset)
        q_next.put((frame, copy.deepcopy(pose_2d_results)))


def my_video_pose_3d(q_pre: Queue, q_next: Queue, q_result: Queue, q_2d: Queue):
    # videopose3D模型定义
    pose_lifter_config = "videopose3D/videopose3d_h36m_27frames_fullconv_semi-supervised_cpn_ft.py"
    pose_lifter_checkpoint = "videopose3D/videopose_h36m_27frames_fullconv_semi-supervised_cpn_ft-71be9cde_20210527.pth"
    pose_lift_model = init_pose_model(pose_lifter_config, pose_lifter_checkpoint, device='cuda:1')

    pose_2d_results = []  # 处理后产生的2D关键点结果列表，用于后续检测(lite -> videopose3d)
    img_2d_num = 0  # 用于记录已经输出的2D关键点的个数(vediopose的局部变量)
    pose_3D_result = []

    # ----------------------------------第二阶段:人体3D关键点估计与动作预测-------------------------------------
    while True:
        frame, pose_2d_results = q_pre.get()
        # for pose_det_results in pose_2d_results:
        #     for res in pose_det_results:
        #         q_2d.put(res['keypoints'])

        # videoPose3D预测
        start_time = time.time()
        img_size = (frame.shape[1], frame.shape[0])
        pose_3D = pose_lift(pose_lift_model, pose_2d_results, img_size)  # 只出一帧
        end1 = time.time() - start_time
        print(f'videopose3D_test time: {end1}')
        pose_3D_result.append(pose_3D[0])

        # 调整videoPose3D的输出使其满足TrajectoryCNN的输入
        pose_3D_ndarry = np.ndarray((len(pose_3D), 17, 3))
        for i, pose in enumerate(pose_3D):
            pose = pose[:, :3]
            pose_3D_ndarry[i] = copy.deepcopy(pose)
        pose_3D_ndarry *= -1000
        q_next.put(pose_3D_ndarry[0])
        q_result.put(copy.deepcopy(pose_3D_ndarry[0])) # 目标9


def my_trajectory_cnn(q: Queue, q_result: Queue):
    # TrajectoryCNN模型定义
    pose3D_predict_model = tcnn.Model()

    # 最终结果
    predict_list = []  # 存储最后的预测结果序列(tcnn最终结果)
    tem_list = []

    i = 0
    time0 = time.time()
    while True:
        pose_3D_ndarry = q.get()
        tem_list.append(pose_3D_ndarry)
        if len(tem_list) >= 10:

            print(
                f"--------------------------------------frame:{i} time:{time.time() - time0}------------------------------")
            time0 = time.time()
            i += 1
            # TraCNN动作预测
            predict_3D = tcnn.TraCNN_predict(pose3D_predict_model, np.array(tem_list))
            # predict_list.append(predict_3D.copy())  # 积攒结果
            q_result.put(predict_3D.copy())

            tem_list.pop(0)


if __name__ == '__main__':
    # main()

    start_multi_process()
