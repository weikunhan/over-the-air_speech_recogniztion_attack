import tensorflow as tf
import ops, sys

def conv2d(inp, shp, name, strides=(1,1,1,1), padding='SAME'):
    with tf.device('/cpu:0'):
        filters = tf.get_variable(name + '/filters', shp, initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name + '/biases', [shp[-1]], initializer=tf.constant_initializer(0.1))
    return tf.nn.bias_add(tf.nn.conv2d(inp, filters, strides=strides, padding=padding), biases)

class UNet(object):

    def __init__(self, batch_loader, devices, optimizer, config):
        self.dic = {}
        self.loss = 0
        outputs = []
        self.config = config
        tower_grads = []
        n_gpus = len(devices)
        x, y = batch_loader.get_batch()
        inputs = tf.split(axis=0, num_or_size_splits=n_gpus, value=x)
        labels = tf.split(axis=0, num_or_size_splits=n_gpus, value=y)
        for i in range(n_gpus):
            with tf.device(devices[i]):
                with tf.variable_scope('UNet'):
                    print(devices[i])
                    try:
                        outputs.append(self.build_tower(inputs[i]))
                    except:
                        tf.get_variable_scope().reuse_variables()
                        outputs.append(self.build_tower(inputs[i]))
                    loss = tf.reduce_mean(tf.square(outputs[-1] - labels[i]))
                    if config.is_training:
                        tower_grads.append(optimizer.compute_gradients(loss))
                        self.train_step = optimizer.apply_gradients(ops.average_gradients(tower_grads))
                    self.loss += loss / n_gpus
                    tf.get_variable_scope().reuse_variables()
        self.output = tf.concat(outputs, axis=0)
        if config.is_training:
            self.train_step = optimizer.apply_gradients(ops.average_gradients(tower_grads))

    def down3(self, inp, in_ch, out_ch, name):
        mid_ch = (in_ch + out_ch) // 2
        conv1 = tf.nn.relu(conv2d(inp, [3,3,in_ch,mid_ch], name + '/conv1'))
        conv2 = tf.nn.relu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2'))
        conv3 = tf.nn.relu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3'))
        tmp = tf.pad(inp, [[0,0], [0,0], [0,0], [0,out_ch-in_ch]], 'CONSTANT')
        self.dic[name] = conv3 + tmp
        return tf.nn.max_pool(self.dic[name], ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')

    def up3(self, inp, in_ch, out_ch, size, name):
        image = tf.image.resize_bilinear(inp, [size, size])
        image = tf.concat([image, self.dic[name.replace('up', 'down')]], axis=3)
        mid_ch = (in_ch + out_ch) // 2
        conv1 = tf.nn.relu(conv2d(image, [3,3,in_ch,mid_ch], name + '/conv1'))
        conv2 = tf.nn.relu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2'))
        conv3 = tf.nn.relu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3'))
        return conv3

    def down(self, inp, in_ch, out_ch, name):
        mid_ch = (in_ch + out_ch) // 2
        conv1 = tf.nn.relu(conv2d(inp, [3,3,in_ch,mid_ch], name + '/conv1'))
        conv2 = tf.nn.relu(conv2d(conv1, [3,3,mid_ch,out_ch], name + '/conv2'))
        tmp = tf.pad(inp, [[0,0], [0,0], [0,0], [0,out_ch-in_ch]], 'CONSTANT')
        self.dic[name] = conv2 + tmp
        return tf.nn.max_pool(self.dic[name], ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')

    def up(self, inp, in_ch, out_ch, size, name):
        image = tf.image.resize_bilinear(inp, [size, size])
        image = tf.concat([image, self.dic[name.replace('up', 'down')]], axis=3)
        mid_ch = (in_ch + out_ch) // 2
        conv1 = tf.nn.relu(conv2d(image, [3,3,in_ch,mid_ch], name + '/conv1'))
        conv2 = tf.nn.relu(conv2d(conv1, [3,3,mid_ch,out_ch], name + '/conv2'))
        return conv2

    def build_tower(self, inp):

        down1 = self.down(inp,   2*3,   32, 'down1')
        down2 = self.down(down1, 32,  64, 'down2')
        down3 = self.down(down2, 64,  128, 'down3')
        down4 = self.down(down3, 128, 256, 'down4')

        ctr = tf.nn.relu(conv2d(down4, [3,3,256,256], 'center'))

        size = self.config.image_size
        up4 = self.up(ctr, 256*2, 128, size//8, 'up4')
        up3 = self.up(up4, 128*2, 64,  size//4, 'up3')
        up2 = self.up(up3, 64*2, 32,  size//2, 'up2')
        up1 = self.up(up2, 32*2, 32,  size, 'up1')

        return conv2d(up1, [3,3,32,6], 'last_layer')
