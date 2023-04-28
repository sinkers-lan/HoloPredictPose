import tensorflow as tf
import os.path
import numpy as np
from TrajectoryCNN.nets import TrajectoryCNN
from utils import metrics
from utils import recoverh36m_3d
from TrajectoryCNN.utils import optimizer
import time
import scipy.io as io
import os, shutil
import pdb
from demo import pose_3D

test_path = 'data/h36m20/my_test/h36m2.npy'
result_path = "results/h36m/v2"
model_path = 'checkpoints/h36m/v1/model.ckpt-769500'
input_length = 10
seq_length = 20
joints_number = 14
joint_dims = 3
stacklength = 4
filter_size = 3
batch_size = 1
n_gpu = 2
num_hidden = [64, 64, 64, 64, 64]
print('!!! TrajectoryCNN:', num_hidden)


class Model(object):
    def __init__(self):
        if model_path:
            print('!!!')
        else:
            print('???')
        # inputs
        self.x = [tf.placeholder(tf.float32, [batch_size,
                                              seq_length + seq_length - input_length,
                                              joints_number,
                                              joint_dims])
                  for i in range(n_gpu)]
        grads = []
        loss_train = []
        self.pred_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        self.keep_prob = tf.placeholder(tf.float32)
        self.params = dict()

        for i in range(n_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(tf.get_variable_scope(),
                                       reuse=True if i > 0 else None):
                    # define a model
                    output_list = TrajectoryCNN.TrajectoryCNN(
                        self.x[i],  # 传入的是一个tf.placeholder
                        self.keep_prob,
                        seq_length,
                        input_length,
                       stacklength,
                        num_hidden,
                        filter_size)

                    gen_ims = output_list[0]
                    loss = output_list[1]
                    pred_ims = gen_ims[:, input_length - seq_length:]#就是和gen_ims一样的东西，不知道为啥要弄两遍!!!
                    loss_train.append(loss)
                    # gradients
                    all_params = tf.trainable_variables()
                    grads.append(tf.gradients(loss, all_params))
                    self.pred_seq.append(pred_ims)#将输出结果添加到队列中

        if n_gpu == 1:
            self.train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)
        else:
            # add losses and gradients together and get training updates
            with tf.device('/gpu:0'):
                for i in range(1, n_gpu):
                    loss_train[0] += loss_train[i]
                    for j in range(len(grads[0])):
                        grads[0][j] += grads[i][j]
            # keep track of moving average
            ema = tf.train.ExponentialMovingAverage(decay=0.9995)
            maintain_averages_op = tf.group(ema.apply(all_params))
            self.train_op = tf.group(optimizer.adam_updates(
                all_params, grads[0], lr=self.tf_lr, mom1=0.95, mom2=0.9995),
                maintain_averages_op)

        self.loss_train = loss_train[0] / n_gpu

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config=configProt)
        self.sess.run(init)
        if model_path:
            print('pretrain model: ', model_path)
            self.saver.restore(self.sess, model_path)

    def train(self, inputs, lr, keep_prob):
        feed_dict = {self.x[i]: inputs[i] for i in range(n_gpu)}
        feed_dict.update({self.tf_lr: lr})
        feed_dict.update({self.keep_prob: keep_prob})
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
        return loss

    def test(self, inputs, keep_prob):
        feed_dict = {self.x[i]: inputs[i] for i in range(n_gpu)}
        feed_dict.update({self.keep_prob: keep_prob})
        gen_ims = self.sess.run(self.pred_seq, feed_dict)
        return gen_ims

    def save(self, itr):
        checkpoint_path = os.path.join("" 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + "")

#测试主函数
def TraCNN_predict(model):
    print('test...')
    # res_path = os.path.join(result_path, 'test')
    # if not tf.gfile.Exists(res_path):
    #     os.mkdir(res_path)
    test_time = 0
    # print('loading inputs from', test_path)
    # data = np.load(test_path)  # (338, 17, 3)

    data = pose_3D

    steps = 1
    n = (len(data) - input_length) // steps + 1
    all_input = np.zeros((len(data), 20, 32, 3)) #错开来看，不够的进行填充
    #左边是下标，右边是关键点编号
    trans_video_pose = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 6,
        5: 7,
        6: 8,
        7: 12,
        8: 13,
        9: 14,
        10: 15,
        11: 17,
        12: 18,
        13: 19,
        14: 25,
        15: 26,
        16: 27
    }
    trans_trajectory_cnn = {
        0: 30, 1: 29, 2: 27, 3: 26, 4: 25, 5: 17, 6: 18, 7: 19, 8: 21, 9: 22, 10: 15, 11: 14, 12: 13, 13: 12, 14: 7,
        15: 8,
        16: 9, 17: 10, 18: 2, 19: 3, 20: 4, 21: 5,
        # 重复点
        22: 16, 23: 20, 24: 23, 25: 24, 26: 28, 27: 31,
        # 不动点
        28: 0, 29: 1, 30: 6, 31: 11
    }
    #关键点编号作为key
    new_trans_video_pose = dict(zip(trans_video_pose.values(), trans_video_pose.keys()))
    #关键点编号作为key
    new_trans_trajectory_cnn = dict(zip(trans_trajectory_cnn.values(), trans_trajectory_cnn.keys()))
    index_list_tem = []
    index_list_data = np.arange(17)
    # 换节点顺序
    for i in range(17):
        #由pose下标快速找到tcnn中的下标
        index_list_tem.append(new_trans_trajectory_cnn[trans_video_pose[i]])
    all_input[:, 0, index_list_tem] = data[:, index_list_data]#完成行的填充

    # 换维度
    for j in range(n):
        for i in range(0, input_length):
            all_input[j, i] = all_input[j * steps + i, 0]
    all_input = np.delete(all_input, range(n + 1, len(data)), axis=0)

    img_gen = np.ndarray((0, seq_length - input_length, joints_number, 3))
    for i in range(int(len(all_input) / batch_size)):
        start_time1 = time.time()
        tem = all_input[i * batch_size:i * batch_size + batch_size]
        tem = np.repeat(tem, n_gpu, axis=0) #卡着重复的
        test_ims = tem[:, 0:seq_length, :, :]
        test_ims = test_ims[:, :, 0:22, :]#取前22个关键点(就已经将0、1、6去除掉了，因为它们是不动点)
        test_ims = np.delete(test_ims, (0, 1, 8, 9, 16, 17, 20, 21), axis=2)#留下14个关键点

        test_dat = test_ims[:, 0:input_length, :, :]#取出有用的前十帧，用于后面拼凑
        tem = test_dat[:, input_length - 1]#都取最后一帧
        tem = np.expand_dims(tem, axis=1)
        tem = np.repeat(tem, seq_length - input_length, axis=1)
        test_dat1 = np.concatenate((test_dat, tem), axis=1)#完成后面多余部分的填充就直接用最后一帧填到满

        test_dat2 = test_ims[:, input_length:]
        test_dat = np.concatenate((test_dat1, test_dat2), axis=1)
        test_dat = np.split(test_dat, n_gpu)
        test_gen = model.test(test_dat, 1)
        end_time1 = time.time()
        t1 = end_time1 - start_time1
        test_time += t1
        # concat outputs of different gpus along batch
        test_gen = np.concatenate(test_gen)
        img_gen = np.concatenate((img_gen, test_gen), axis=0)

    print(f'test time: {test_time}')

    # 换维度
    save_data_1 = np.zeros(((n - n % batch_size - 1) * steps + input_length, joints_number, 3))
    for ik in range(n - n % batch_size):
        for step in range(steps):
            save_data_1[ik * steps + step] = img_gen[
                ik * n_gpu, (seq_length - input_length - steps) // 2 + step]
    # static_joints = np.array(
    #     [[[-1.32948593e+02, 0.00000000e+00, 0.00000000e+00],
    #      [1.32948822e+02, 0.00000000e+00, 0.00000000e+00]]]
    # )

    # 换结点顺序
    index_before = np.array([2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 18, 19])#tcnn
    index_after = []
    for i in index_before:
        index_after.append(new_trans_video_pose[trans_trajectory_cnn[i]])
    save_data_2 = np.zeros((len(save_data_1), 17, 3))
    save_data_2[:, index_after] = save_data_1[:, np.arange(14)]

    # save_data_2[:, [1, 4]] = np.repeat(static_joints, len(save_data_2), axis=0)
    save_data_2[:, [1, 4]] = data[:save_data_2.shape[0], [1, 4]]
    save_data_2 = np.insert(save_data_2, np.zeros((10, ), dtype=np.intp), values=np.zeros(save_data_2.shape[1:]), axis=0)

    # save prediction examples
    save_data_3 = save_data_2.reshape(-1)
    save_data_3 = save_data_3 / (np.max(save_data_3) - np.min(save_data_3)) * 2
    save_data_3 = save_data_3.reshape((-1, 17, 3))

    return save_data_3
    # path = res_path
    # if not tf.gfile.Exists(path):
    #     os.mkdir(path)
    # np.save(os.path.join(path, 'all_input.npy'), all_input)
    # np.save(path, save_data_3)
    # np.save(os.path.join(path, 'img_gen.npy'), img_gen)
    # print('save test done!')


def main(argv=None):
    if not tf.gfile.Exists(result_path):
        tf.gfile.MakeDirs(result_path)

    print('start training !', time.strftime('%Y-%m-%d %H:%M:%S\n\n\n', time.localtime(time.time())))

    print('Initializing models')
    model = Model()
    TraCNN_predict(model)



if __name__ == '__main__':
    main()
