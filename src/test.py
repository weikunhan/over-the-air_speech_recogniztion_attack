from configobj import ConfigObj
import tensorflow as tf
from tqdm import tqdm
import glob, os, sys
import network, ops
import batch_utils
import numpy as np
import scipy.io

if __name__ == '__main__':

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        images = sorted(glob.glob('D:/Ryan/U_Net_Sub/data/test/input/*.npy'))
        config = ConfigObj()
        config.batch_size = 1
        config.image_size = 512
        config.is_training = False
        config.q_limit = 10
        config.TTA = False

        input_ = tf.placeholder(tf.float32, shape=[None, 512, 512, 6])
        label_ = tf.placeholder(tf.float32, shape=[None, 512, 512, 6])
        batch_loader = batch_utils.TestBatchLoader(images, input_, label_, config)
        devices = ops.get_available_gpus()[:1]
        infer_net = network.UNet(batch_loader, devices, None, config)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            tf.train.Saver().restore(sess, 'Models/best_model')
            tf.train.start_queue_runners(sess=sess)
            batch_loader.start_threads(sess, 1)
            os.system('rm -r outputs')
            os.system('mkdir outputs')
            for i in tqdm(range(len(images))):
                path = 'outputs/{}'.format(images[i].split('\\')[-1].split('.')[0])
                x = np.load(images[i])
                y = np.load(images[i].replace('input', 'target'))
                scipy.io.savemat(path, {'input': x, 'target': y, 'output': sess.run(infer_net.output)})
