import math
import matplotlib.pyplot as plt
import tensorflow as tf

# from PIL import Image.ImageDraw

def instance_norm(input):
    epsilon = 1e-9
    mean, var = tf.nn.moments(input, [1, 2], keep_dims=True)

    return tf.div(tf.sub(input, mean), tf.sqrt(tf.add(var, epsilon)))

def resnet(x, filters, kernel, strides, name='resnet', b_name='b_name',reuse=False):
    with tf.variable_scope(name) as scope:
        conv1 = tf.nn.relu(instance_norm(conv2d_(x, filters, filters, kernel, strides, padding=1, name='conv1')))
        conv2 = instance_norm(conv2d_(conv1, filters, filters, kernel, strides, padding=1, name='conv2'))
        residual = x + conv2
        return residual

def deconv2d(x, in_size, out_size, kernel, strides, p='SAME', name='conv_transpose'):
    with tf.variable_scope(name) as scope:
        shape = [kernel,kernel,out_size,in_size]
        W = tf.get_variable('w', shape, initializer=tf.random_normal_initializer(stddev=0.02))
        strides = [1,strides,strides,1]

        _,_,c,_ = W.get_shape().as_list()
        b,h,w,_ = x.get_shape().as_list()
        return tf.nn.conv2d_transpose(x, W, [b, strides[1] * h, strides[1] * w, c], strides=strides, padding=p, name=name)

def conv2d_(x, input_filters, output_filters, kernel, strides, padding = 1, mode='CONSTANT', name='conv'):
    with tf.variable_scope(name) as scope:
        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding='SAME', name=name)

        return conv

def leaky_relu(input, val):
    with tf.name_scope("leaky"):
        x = tf.identity(input)
        return (0.5 * (1 + val)) * x + (0.5 * (1 - val)) * tf.abs(x)

def batchnorm(input,name="batchnorm"):
    with tf.variable_scope(name):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized
