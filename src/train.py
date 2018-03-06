from configobj import ConfigObj
from time import time, sleep
import batch_utils, network
import tensorflow as tf
from tqdm import tqdm
import glob, ops, sys
import numpy as np
import random

def init_parameters():
    tc, vc = ConfigObj(), ConfigObj()
    tc.is_training, vc.is_training = True, False
    tc.image_size, vc.image_size = 256, 512
    tc.batch_size, vc.batch_size = 64, 2
    tc.n_threads, vc.n_threads = 10, 1
    tc.checkpoint = 500
    tc.q_limit, vc.q_limit = 10000, 50
    return tc, vc

if __name__ == '__main__':

    train_images = glob.glob('D:/Ryan/U_Net_Sub/data/train/input/*.npy')
    random.shuffle(train_images)
    valid_images = glob.glob('D:/Ryan/U_Net_Sub/data/valid/input/*.npy')

    train_config, valid_config = init_parameters()
    s1, s2 = train_config.image_size, valid_config.image_size

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        train_input_ = tf.placeholder(tf.float32, shape=[None, s1, s1, 6])
        train_label_ = tf.placeholder(tf.float32, shape=[None, s1, s1, 6])
        train_bl = batch_utils.TrainBatchLoader(train_images, train_input_, train_label_, train_config)

        valid_input_ = tf.placeholder(tf.float32, shape=[None, s2, s2, 6])
        valid_label_ = tf.placeholder(tf.float32, shape=[None, s2, s2, 6])
        valid_bl = batch_utils.ValidBatchLoader(valid_images, valid_input_, valid_label_, valid_config)

        optimizer = tf.train.AdamOptimizer(1e-4)
        devices = ops.get_available_gpus()
        train_net = network.UNet(train_bl, devices, optimizer, train_config)
        valid_net = network.UNet(valid_bl, devices, optimizer, valid_config)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            tf.train.start_queue_runners(sess=sess)
            train_bl.start_threads(sess, n_threads=train_config.n_threads)
            valid_bl.start_threads(sess, n_threads=valid_config.n_threads)
            sleep(20)
            print(train_bl.queue.size().eval(), valid_bl.queue.size().eval())

            train_log = open('train_log.txt', 'w')
            valid_log = open('valid_log.txt', 'w')

            n_eval_steps = (len(valid_images) + valid_config.batch_size - 1) // valid_config.batch_size
            check = train_config.checkpoint
            train_loss = 0
            min_loss = float('inf')
            start_time = time()

            for i in range(1, 1000000):
                t = time()
                _, a = sess.run([train_net.train_step, train_net.loss])
                format_str = ('iter: %d loss: %.4f backprop: %.2f time: %d')
                text = (format_str % (i, a, time() - t, int(time()-start_time)))
                ops.print_out(train_log, text)
                train_loss += a

                if i % check == 0:
                    res = ops.evaluate(sess, n_eval_steps, [valid_net.loss])
                    format_str = ('iter: %d loss: %.4f train_loss: %.4f time: %d')
                    text = (format_str % (i, res[0], train_loss/check, int(time()-start_time)))
                    ops.print_out(valid_log, text)
                    train_loss = 0
                    if res[0] < min_loss:
                        min_loss = res[0]
                        saver.save(sess, 'Models/best_model')
