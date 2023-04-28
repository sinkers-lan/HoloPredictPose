import numpy as np
import pdb
# 此函数与计算损失值有关
def recoverh36m_3d(gt,pred):
	joint_to_ignore = np.array([22, 23, 24, 25, 26, 27])
	joint_equal = np.array([12, 7, 9, 12, 2, 0])
	"""
	------------------------------------改动4-----------------------------------------------------------------------
	添加了一行
	joint_not_used = np.array([0, 1, 8, 9, 16, 17, 20, 21])
	"""
	joint_not_used = np.array([0, 1, 8, 9, 16, 17, 20, 21])
	unchange_joint=np.array([28,29,30,31])  # corresponding to original joints: 0,1,6,11
	tem=np.zeros([gt.shape[0],gt.shape[1],len(joint_to_ignore)+len(unchange_joint),gt.shape[-1]])
	#pdb.set_trace()
	pred_3d=np.concatenate((pred, tem), axis=2)
	"""
	------------------------------------改动5-----------------------------------------------------------------------
	添加了2行
	for index in joint_not_used:
		pred_3d = np.insert(pred_3d, index, values=gt[:, :, index], axis=2)
	此处修改是因为计算损失和评估指标的时候会调用这个函数
	"""
	for index in joint_not_used:
		pred_3d = np.insert(pred_3d, index, values=gt[:, :, index], axis=2)
	# pred_3d = np.insert(pred_3d, joint_not_used, values=gt[:, :, joint_not_used], axis=2)
	pred_3d[:,:,joint_to_ignore]=pred_3d[:,:,joint_equal]  # 将相同的位置用相同位置的点补上
	pred_3d[:,:,unchange_joint]=gt[:,:,unchange_joint]  # 将未变化的点用gt补上

	return pred_3d
