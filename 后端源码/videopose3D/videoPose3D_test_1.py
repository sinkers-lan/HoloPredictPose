import matplotlib.pyplot as plt
import copy
import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np
import torch

from mmpose.apis import (extract_pose_sequence,
                         get_track_id, inference_pose_lifter_model,
                         inference_top_down_pose_model, init_pose_model, vis_pose_result,
                         process_mmdet_results, vis_3d_pose_result)
from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo
from mmpose.models import PoseLifter, TopDown


# from demo import pose_2d_results,img_size

# from demo import pose_lifter_config
# from demo import pose_lifter_checkpoint
# from demo import pose_lift_model


def pose_lift(pose_lift_model, pose_2d_results, img_size=None):
    # 一些设置
    rebase_keypoint_height = True

    print('Stage 2: 2D-to-3D pose lifting.')

    assert isinstance(pose_lift_model, PoseLifter), \
        'Only "PoseLifter" model is supported for the 2nd stage ' \
        '(2D-to-3D lifting)'

    pose_lift_dataset = pose_lift_model.cfg.data['test']['type']

    # if args.out_video_root == '':
    #     save_out_video = False
    # else:
    #     os.makedirs(args.out_video_root, exist_ok=True)
    #     save_out_video = True

    # if save_out_video:
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     fps = video.fps
    #     writer = None

    # load temporal padding config from model.data_cfg
    if hasattr(pose_lift_model.cfg, 'test_data_cfg'):
        data_cfg = pose_lift_model.cfg.test_data_cfg
    else:
        data_cfg = pose_lift_model.cfg.data_cfg

    # build pose smoother for temporal refinement
    # if args.smooth:
    #     smoother = Smoother(
    #         filter_cfg=args.smooth_filter_cfg,
    #         keypoint_key='keypoints',
    #         keypoint_dim=2)
    # else:
    #     smoother = None

    num_instances = -1
    pose_lift_dataset_info = pose_lift_model.cfg.data['test'].get(
        'dataset_info', None)
    if pose_lift_dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        pose_lift_dataset_info = DatasetInfo(pose_lift_dataset_info)

    print('Running 2D-to-3D pose lifting inference...')
    returned_pose_lift = []

    save_cut = []
    save_result = []

    # for i, pose_det_results in enumerate(
    #         pose_2d_results):  # mmcv.track_iter_progress是为了画进度条而封装了一层
    # print(f"causal = {data_cfg.causal}")
    # extract and pad input pose2d sequence
    pose_results_2d = extract_pose_sequence(
        pose_2d_results,
        frame_idx=26,  # 只对最后一帧做预测
        causal=data_cfg.causal,
        seq_len=data_cfg.seq_len,
        step=data_cfg.seq_frame_interval)

    # save_cut.append([i[0]['keypoints'] for i in pose_results_2d])  # (10, 27, 17, 3)

    # 2D-to-3D pose lifting
    pose_lift_results = inference_pose_lifter_model(
        pose_lift_model,
        pose_results_2d=pose_results_2d,
        dataset=pose_lift_dataset,
        dataset_info=pose_lift_dataset_info,
        with_track_id=True,
        image_size=img_size)

    # test_out = pose_lift_results[0]["keypoints_3d"]
    # x = test_out[:, 0]
    # y = test_out[:, 1]
    # z = test_out[:, 2]
    # plt.scatter(x, y)
    # # plt.scatter(y, z)
    # list1 = [10, 9, 8, 7, 0]
    # plt.plot(x[list1], y[list1])
    # list2 = [16, 15, 14, 8, 11, 12, 13]
    # plt.plot(x[list2], y[list2])
    # list3 = [3, 2, 1, 0, 4, 5, 6]
    # plt.plot(x[list3], y[list3])
    # plt.show()
    # save_result.append(pose_lift_results[0]["keypoints_3d"])  # (10, 17, 4)

    returned_pose_lift.append(copy.deepcopy(pose_lift_results[0]["keypoints_3d"]))

    # np.save("outputs/cut.npy", np.array(save_cut))
    # np.save("outputs/cut_result.npy", np.array(save_result))
    return returned_pose_lift

    #     # Pose processing
    #     pose_lift_results_vis = []
    #     for idx, res in enumerate(pose_lift_results):
    #         keypoints_3d = res['keypoints_3d']
    #         # exchange y,z-axis, and then reverse the direction of x,z-axis
    #         keypoints_3d = keypoints_3d[..., [0, 2, 1]]
    #         keypoints_3d[..., 0] = -keypoints_3d[..., 0]
    #         keypoints_3d[..., 2] = -keypoints_3d[..., 2]
    #         # rebase height (z-axis)
    #         if rebase_keypoint_height:
    #             keypoints_3d[..., 2] -= np.min(
    #                 keypoints_3d[..., 2], axis=-1, keepdims=True)
    #         res['keypoints_3d'] = keypoints_3d
    #         # add title
    #         det_res = pose_det_results[idx]
    #         instance_id = det_res['track_id']
    #         res['title'] = f'Prediction ({instance_id})'
    #         # only visualize the target frame
    #         res['keypoints'] = det_res['keypoints']
    #         res['bbox'] = det_res['bbox']
    #         res['track_id'] = instance_id
    #         pose_lift_results_vis.append(res)
    #
    #     # Visualization
    #     if num_instances < 0:
    #         num_instances = len(pose_lift_results_vis)
    #     img_vis = vis_3d_pose_result(
    #         pose_lift_model,
    #         result=pose_lift_results_vis,
    #         img=video[i],
    #         dataset=pose_lift_dataset,
    #         dataset_info=pose_lift_dataset_info,
    #         out_file=None,
    #         radius=args.radius,
    #         thickness=args.thickness,
    #         num_instances=num_instances,
    #         show=args.show)
    #
    #     if save_out_video:
    #         if writer is None:
    #             writer = cv2.VideoWriter(
    #                 osp.join(args.out_video_root,
    #                          f'vis_{osp.basename(args.video_path)}'), fourcc,
    #                 fps, (img_vis.shape[1], img_vis.shape[0]))
    #         writer.write(img_vis)
    #
    # if save_out_video:
    #     writer.release()


if __name__ == '__main__':
    pose_lift('a', 'a')
