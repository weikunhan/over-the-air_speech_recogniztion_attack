# -*- coding: utf-8 -*-
""" Model

This module can use for build audio U-Net as the libary. You don't need modify
any functions in the libary. Please see the description in front of each
function if you don't undersand it. Please import this module if you want use
this libary in anthor project.

################################################################################
# Author: Weikun Han <weikunhan@gmail.com>
# Crate Date: 03/6/2018
# Update:
# Reference: https://github.com/jhetherly/EnglishSpeechUpsampler
################################################################################
"""

import tensorflow as tf

#custom_shuffle_module = tf.load_op_library('src/shuffle_op.so')
#shuffle = custom_shuffle_module.shuffle

#############################
# TENSORBOARD HELPER FUNCTION
#############################

def comprehensive_variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def histogram_variable_summaries(var):
    """
    Attach a histogram summary to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        tf.summary.histogram('histogram', var)

########################
# LAYER HELPER FUNCTIONS
########################

def subpixel_reshuffle_1D_impl(X, m):
    """
    performs a 1-D subpixel reshuffle of the input 2-D tensor
    assumes the last dimension of X is the filter dimension
    ref: https://github.com/Tetrachrome/subpixel
    """
    return tf.transpose(tf.stack([tf.reshape(x, (-1,)) for x
                                  in tf.split(X, m, axis=1)]))


def subpixel_reshuffle_1D(X, m, name=None):
    """
    maps over the batch dimension
    """
    return tf.map_fn(lambda x: subpixel_reshuffle_1D_impl(x, m), X, name=name)


def subpixel_restack_impl(X, n_prime, m_prime, name=None):
    """
    performs a subpixel restacking such that it restacks columns of a 2-D
    tensor onto the rows
    """
    bsize = tf.shape(X)[0]
    r_n = n_prime - X.get_shape().as_list()[1]
    total_new_space = r_n*m_prime
    to_stack = tf.slice(X, [0, 0, m_prime], [-1, -1, -1])
    to_stack = tf.slice(tf.reshape(to_stack, (bsize, -1)),
                        [0, 0], [-1, total_new_space])
    to_stack = tf.reshape(to_stack, (bsize, -1, m_prime))
    to_stack = tf.slice(to_stack, [0, 0, 0], [-1, r_n, -1])
    return tf.concat((tf.slice(X, [0, 0, 0], [-1, -1, m_prime]), to_stack),
                     axis=1, name=name)


def subpixel_restack(X, n_prime, m_prime=None, name=None):
    n = X.get_shape().as_list()[1]
    m = X.get_shape().as_list()[2]
    r_n = n_prime - n
    if m_prime is None:
        for i in range(1, m):
            r_m = i
            m_prime = m - r_m
            if r_m*n >= m_prime*r_n:
                break
    return subpixel_restack_impl(X, n_prime, m_prime, name=name)


def batch_normalization(input_tensor, is_training, scope):
    """ Build general batch normalizaion function

    This function is use for add batch normalization for deep neural netwok.
    In the neural network training, when the convergence speed is very slow, or
    when a gradient explosion or other untrainable condition is encountered,
    batch normalizaion can be tried to solve. In addition, batch normalizaion
    can also be added under normal usage to speed up training and
    improve model accuracy.

    Args:
        param1 (tensor): input_tensor
        param2 (bool): is_training
        param3 (str): scope

    Returns:
        tensor: output layer add the batch normaliazation function

    """

    # Selet batch nomalization is use for training or not traning
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(input_tensor,
                                                        decay=0.99,
                                                        is_training=is_training,
                                                        center=True,
                                                        scale=True,
                                                        updates_collections=None,
                                                        scope=scope,
                                                        reuse=False),
                   lambda: tf.contrib.layers.batch_norm(input_tensor,
                                                        decay=0.99,
                                                        is_training=is_training,
                                                        center=True,
                                                        scale=True,
                                                        updates_collections=None,
                                                        scope=scope,
                                                        reuse=False))

def weight_variables(shape, name=None):
    """ Build normal distributions generator

    Output a random value from the truncated normal distribution.
    The resulting value follows a normal distribution with a specified mean
    and standard deviation, and discards the reselection if the resulting value
    is greater than the value of the mean 2 standard deviations.

    Args:
        param1 (list): shape
        param2 (str): name

    Returns:
        list: the tensorfolw veriable

    """

    # Use truncated_normal or random_normal to build
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variables(shape, name=None):
    """ Build constant generator

    This function add the initial bias 0.1 for each depth of tensor
    Output a constant variable

    Args:
        param1 (list): shape
        param2 (str): name

    Returns:
        list: the tensorfolw veriable

    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

########################
# SINGLE LAYER FUNCTIONS
########################

def convolution_1d_act(prev_tensor,
                       prev_num_filters,
                       filter_size,
                       num_filters,
                       active_function,
                       layer_number,
                       stride=1,
                       padding='SAME',
                       tensorboard_output=False,
                       name=None):
    """ Build a single convolution layer with active function

    The function build a single convolution layer with intial weight and bias.
    Also add the active function for this single covolution layer

    Args:
        param1 (tensor): pre_tensor
        param2 (int): prev_num_filters
        param3 (int): filter_size
        param4 (int): num_filters
        param5 (funtion): active_function
        param6 (int): layer_number
        param7 (int): stride
        param8 (str): padding
        param9 (bool): tensorboard_output
        param10 (str): name

    Returns:
        tensor: representing the output of the operation

    """

    # Define the filter
    with tf.name_scope('{}_layer_conv_weights'.format(layer_number)):
        w = weight_variable(
            [filter_size, prev_num_filters, num_filters])

        if tensorboard_output:
            histogram_variable_summaries(w)

    # Define the bias
    with tf.name_scope('{}_layer_conv_biases'.format(layer_number)):
        b = bias_variable([num_filters])

        if tensorboard_output:
            histogram_variable_summaries(b)

    # Create the single convolution laryer
    with tf.name_scope('{}_layer_conv_preactivation'.format(layer_number)):
        conv = tf.nn.conv1d(prev_tensor, w, stride=stride, padding=padding) + b

        if tensorboard_output:
            histogram_variable_summaries(conv)

    # Add the active function
    with tf.name_scope('{}_layer_conv_activation'.format(layer_number)):
        conv_act = active_function(conv, name=name)

        if tensorboard_output:
            histogram_variable_summaries(conv_act)
    return conv_act

def downsampling_d1_batch_norm_act(input_tensor,
                                   filter_size,
                                   stride,
                                   layer_number,
                                   active_function=tf.nn.relu,
                                   is_training=True,
                                   num_filters=None,
                                   padding='VALID',
                                   tensorboard_output=False,
                                   name=None):
    """ Build a single downsampling layer

    The function build a single convolution layer with intial weight and bias.
    Also add  the batch normalization for this single convolution layer.
    Also add the active function for this single covolution layer.

    Args:
        param1 (tensor): input_tensor
        param2 (int): filter_size
        param3 (int): stride
        param4 (int): layer_number
        param5 (funtion): active_function
        param6 (bool): is_training
        param7 (int): num_filters
        param8 (str): padding
        param9 (bool): tensorboard_output
        param10 (str): name

    Returns:
        tensor: representing the output of the operation

    """

    # Assume this layer is twice the depth of the previous layer if no depth
    # information is given
    if num_filters is None:
        num_filters = 2 * input_tensor.get_shape().as_list()[-1]

    # Define the filter
    with tf.name_scope('{}_layer_conv_weights'.format(layer_number)):

        # input_tensor.get_shape().as_list()[-1] is pre_num_fitlters
        w = weight_variable(
            [filter_size, input_tensor.get_shape().as_list()[-1], num_filters])

        if tensorboard_output:
            histogram_variable_summaries(w)

    # Define the bias
    with tf.name_scope('{}_layer_conv_biases'.format(layer_number)):
        b = bias_variable([num_filters])

        if tensorboard_output:
            histogram_variable_summaries(b)

    # Create the single convolution laryer
    with tf.name_scope('{}_layer_conv_preactivation'.format(layer_number)):
        conv = tf.nn.conv1d(input_tensor, w, stride=stride, padding=padding) + b

        if tensorboard_output:
            histogram_variable_summaries(conv)

    # Add the batch nomalization at output of conlution laryer
    with tf.name_scope('{}_layer_batch_norm'.format(layer_number)) as scope:
        conv_batch_norm = batch_normalization(conv, is_training, scope)

    # Add the active function
    with tf.name_scope('{}_layer_conv_activation'.format(layer_number)):
        conv_batch_norm_act = active_function(conv_batch_norm, name=name)

        if tensorboard_output:
            histogram_variable_summaries(conv_batch_norm_act)
    return conv_batch_normal_act


def upsampling_d1_batch_normal_act_subpixel(input_tensor,
                                            residual_tensor,
                                            filter_size,
                                            stride=1
                                            layer_number,
                                            active_function=tf.nn.relu,
                                            is_training=True,
                                            num_filters=None,
                                            padding='VALID',
                                            tensorboard_output=False,
                                            name=None):
    """ Build a single upsampling layer

    The function build a single convolution layer with intial weight and bias.
    Also add  the batch normalization for this single convolution layer.
    Also add the active function for this single covolution layer.
    Also a subpixel convolution that reorders information along one
    dimension to expand the other dimensions.
    Also final convolutional layer with restacking and reordering operations
    is residually added to the original input to yield the upsampled waveform.

    Args:
        param1 (tensor): input_tensor
        param2 (tensor): residual_tensor
        param3 (int): filter_size
        param4 (int): stride
        param5 (int): layer_number
        param6 (funtion): active_function
        param7 (bool): is_training
        param8 (int): num_filters
        param9 (str): padding
        param10 (bool): tensorboard_output
        param11 (str): name

    Returns:
        tensor: representing the output of the operation

    """

    # assume this layer is half the depth of the previous layer if no depth
    # information is given
    if num_filters is None:
        num_filters = int(input_tensor.get_shape().as_list()[-1] / 2)

    # Define the filter
    with tf.name_scope('{}_layer_conv_weights'.format(layer_number)):

        # input_tensor.get_shape().as_list()[-1] is pre_num_fitlters
        w = weight_variable(
            [filter_size, input_tensor.get_shape().as_list()[-1], num_filters])

        if tensorboard_output:
            histogram_variable_summaries(w)

    # Define the bias
    with tf.name_scope('{}_layer_conv_biases'.format(layer_number)):
        b = bias_variable([num_filters])

        if tensorboard_output:
            histogram_variable_summaries(b)

    # Create the single convolution laryer
    with tf.name_scope('{}_layer_conv_preactivation'.format(layer_number)):
        conv = tf.nn.conv1d(input_tensor, w, stride=stride, padding=padding) + b

        if tensorboard_output:
            histogram_variable_summaries(conv)

    # Add the batch nomalization at output of conlution laryer
    with tf.name_scope('{}_layer_batch_norm'.format(layer_number)) as scope:
        conv_batch_norm = batch_normalization(l, is_training, scope)

    # Add the active function
    with tf.name_scope('{}_layer_conv_activation'.format(layer_number)):
        conv_batch_norm_act = active_function(conv_batch_norm, name=name)

        if tensorboard_output:
            histogram_variable_summaries(conv_batch_norm_act)

    # Build a subpixel convolution
    with tf.name_scope('{}_layer_subpixel_reshuffle'.format(layer_number)):
        subpixel_conv = subpixel_reshuffle_1d(
            conv_batch_norm_act,
            residual_tensor.get_shape().as_list()[-1],
            name=name)

        if tensorboard_output:
            histogram_variable_summaries(subpixel_conv)

    # Combined the subpixel convolution restack
    with tf.name_scope('{}_layer_stacking'.format(layer_number)):
        sliced = tf.slice(residual_tensor,
                          begin=[0, 0, 0],
                          size=[-1, subpixel_conv.get_shape().as_list()[1], -1])
        sliced_subpixel_conv = tf.concat((subpixel_conv, sliced),
                                         axis=2,
                                         name=name)

        if tensorboard_output:
            histogram_variable_summaries(sliced_subpixel_conv)

    return sliced_subpixel_conv

#################################
# AUDIO U-NET DEEP NEURAL NETWORK
#################################

def audio_u_net_dnn(input_type,
                    input_shape,
                    num_downsample_layers=8,
                    channel_multiple=8,
                    initial_filter_size=5,
                    initial_stride=2,
                    downsample_filter_size=3,
                    downsample_stride=2,
                    bottleneck_filter_size=4,
                    bottleneck_stride=2,
                    upsample_filter_size=3,
                    tensorboard_output=True,
                    scope_name='audio_u_net_dnn'):
    """ Construct the deep neural network (U-Net)

    The audio U-Net is based on the paper
    https://arxiv.org/abs/1708.00853

    Args:
        param1 (data type): input_type
        param2 (list): input_shape
        param3 (int):
        param4 (int):
        param5 (int):
        param6 (int):
        param7 (int):
        param8 (int):
        param9 (int):
        param10 (bool): tensorboard_output
        param11 (str): name

    Returns:
        tensor: representing the output of the operation

    """

    print('The network summary for {}'.format(scope_name))

    # Create list to store layers
    downsample_layers = []
    upsample_layers = []

    with tf.name_scope(scope_name):

        # is_training variable
        train_flag = tf.placeholder(tf.bool)

        # input of the model (examples)
        s = [None]

        for n in input_shape:
            s.append(i)

        x = tf.placeholder(input_type, shape=s)
        input_size = s[-2]
        num_of_channels = s[-1]

        print(' input: {}'.format(x.get_shape().as_list()[1:]))

        d1 = build_downsampling_block(x,
                                      filter_size=initial_filter_window,
                                      stride=initial_stride,
                                      tensorboard_output=tensorboard_output,
                                      depth=channel_multiple*num_of_channels,
                                      is_training=train_flag,
                                      layer_number=1)

        print(' downsample layer: {}'.format(d1.get_shape().as_list()[1:]))

        downsample_layers.append(d1)

        layer_count = 2
        for i in range(number_of_downsample_layers - 1):
            d = build_downsampling_block(
                downsample_layers[-1],
                filter_size=downsample_filter_window,
                stride=downsample_stride,
                tensorboard_output=tensorboard_output,
                is_training=train_flag,
                layer_number=layer_count)
            print(' downsample layer: {}'.format(d.get_shape().as_list()[1:]))
            downsample_layers.append(d)
            layer_count += 1

        bn = build_downsampling_block(downsample_layers[-1],
                                      filter_size=bottleneck_filter_window,
                                      stride=bottleneck_stride,
                                      tensorboard_output=tensorboard_output,
                                      is_training=train_flag,
                                      layer_number=layer_count)
        print(' bottleneck layer: {}'.format(bn.get_shape().as_list()[1:]))
        layer_count += 1

        u1 = build_upsampling_block(bn, downsample_layers[-1],
                                    depth=bn.get_shape().as_list()[-1],
                                    filter_size=upsample_filter_window,
                                    tensorboard_output=tensorboard_output,
                                    is_training=train_flag,
                                    layer_number=layer_count)
        print(' upsample layer: {}'.format(u1.get_shape().as_list()[1:]))
        upsample_layers.append(u1)
        layer_count += 1

        for i in range(number_of_downsample_layers - 2, -1, -1):
            u = build_upsampling_block(upsample_layers[-1],
                                       downsample_layers[i],
                                       filter_size=upsample_filter_window,
                                       tensorboard_output=tensorboard_output,
                                       is_training=train_flag,
                                       layer_number=layer_count)
            print(' upsample layer: {}'.format(u.get_shape().as_list()[1:]))
            upsample_layers.append(u)
            layer_count += 1

        target_size = int(input_size/initial_stride)
        restack = subpixel_restack(upsample_layers[-1],
                                   target_size + (upsample_filter_window - 1))
        print(' restack layer: {}'.format(restack.get_shape().as_list()[1:]))

        conv = build_1d_conv_layer(restack, restack.get_shape().as_list()[-1],
                                   upsample_filter_window, initial_stride,
                                   tf.nn.elu, layer_count,
                                   padding='VALID',
                                   tensorboard_output=tensorboard_output)
        print(' final convolution layer: {}'.format(
            conv.get_shape().as_list()[1:]))

        # NOTE this effectively is a linear activation on the last conv layer
        y = subpixel_reshuffle_1D(conv,
                                  num_of_channels)
        y = tf.add(y, x, name=scope_name)
        print(' output: {}'.format(y.get_shape().as_list()[1:]))

    return train_flag, x, y
