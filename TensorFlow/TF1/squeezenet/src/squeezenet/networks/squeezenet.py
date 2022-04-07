from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d
from tensorflow.contrib.layers import batch_norm, l2_regularizer
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope


@add_arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                data_format='NCHW'):
    with tf.compat.v1.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with arg_scope([conv2d, max_pool2d]):
            net = _squeeze(inputs, squeeze_depth)
            net = _expand(net, expand_depth, data_format)
        return net


def _squeeze(inputs, num_outputs):
    return conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')


def _expand(inputs, num_outputs, data_format):
    concat_axis = 1 if data_format == 'NCHW' else 3

    with tf.compat.v1.variable_scope('expand'):
        e1x1 = conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = conv2d(inputs, num_outputs, [3, 3], scope='3x3')

    return tf.concat([e1x1, e3x3], concat_axis)


class Squeezenet(object):
    """Original squeezenet architecture for 224x224 images."""
    name = 'squeezenet'

    def __init__(self, args):
        self._num_classes = args.num_classes
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False
        self._data_format = args.data_format

    def build(self, x, is_training):
        self._is_built = True
        with tf.variable_scope(self.name, values=[x]):
            with arg_scope(_arg_scope(is_training,
                                      self._weight_decay,
                                      self._batch_norm_decay,
                                      self._data_format)):
                return self._squeezenet(x, self._num_classes, self._data_format)

    @staticmethod
    def _squeezenet(images, num_classes=1000, data_format='NCHW'):
        net = conv2d(images, 96, [2, 2], stride=2, scope='conv1')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
        net = fire_module(net, 16, 64, scope='fire2', data_format=data_format)
        net = fire_module(net, 16, 64, scope='fire3', data_format=data_format)
        net = fire_module(net, 32, 128, scope='fire4', data_format=data_format)
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool4')
        net = fire_module(net, 32, 128, scope='fire5', data_format=data_format)
        net = fire_module(net, 48, 192, scope='fire6', data_format=data_format)
        net = fire_module(net, 48, 192, scope='fire7', data_format=data_format)
        net = fire_module(net, 64, 256, scope='fire8', data_format=data_format)
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
        net = fire_module(net, 64, 256, scope='fire9', data_format=data_format)
        net = conv2d(net, num_classes, [1, 1], stride=1, scope='conv10')
        net = avg_pool2d(net, [13, 13], stride=1, scope='avgpool10')

        squeeze_axes = [2, 3] if data_format == 'NCHW' else [1, 2]
        logits = tf.squeeze(net, squeeze_axes, name='logits')
        return logits


class Squeezenet_CIFAR(object):
    """Modified version of squeezenet for CIFAR images"""
    name = 'squeezenet_cifar'

    def __init__(self, args):
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False
        self._data_format = args.data_format

    def build(self, x, is_training):
        self._is_built = True
        with tf.compat.v1.variable_scope(self.name, values=[x]):
            with arg_scope(_arg_scope(is_training,
                                      self._weight_decay,
                                      self._batch_norm_decay,
                                      self._data_format)):
                return self._squeezenet(x, data_format=self._data_format)

    @staticmethod
    def _squeezenet(images, num_classes=10, data_format='NCHW'):
        net = conv2d(images, 96, [2, 2], scope='conv1')
        net = max_pool2d(net, [2, 2], scope='maxpool1')
        net = fire_module(net, 16, 64, scope='fire2', data_format=data_format)
        net = fire_module(net, 16, 64, scope='fire3', data_format=data_format)
        net = fire_module(net, 32, 128, scope='fire4', data_format=data_format)
        net = max_pool2d(net, [2, 2], scope='maxpool4')
        net = fire_module(net, 32, 128, scope='fire5', data_format=data_format)
        net = fire_module(net, 48, 192, scope='fire6', data_format=data_format)
        net = fire_module(net, 48, 192, scope='fire7', data_format=data_format)
        net = fire_module(net, 64, 256, scope='fire8', data_format=data_format)
        net = max_pool2d(net, [2, 2], scope='maxpool8')
        net = fire_module(net, 64, 256, scope='fire9', data_format=data_format)
        net = avg_pool2d(net, [4, 4], scope='avgpool10')
        net = conv2d(net, num_classes, [1, 1],
                     activation_fn=None,
                     normalizer_fn=None,
                     scope='conv10')

        squeeze_axes = [2, 3] if data_format == 'NCHW' else [1, 2]
        logits = tf.squeeze(net, squeeze_axes, name='logits')
        return logits


class Squeezenet_Tiny(object):
    """Modified version of squeezenet for Tiny ImageNet images."""
    name = 'squeezenet_tiny'

    def __init__(self, args):
        self._num_classes = args.num_classes
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False
        self._data_format = args.data_format

    def build(self, x, is_training):
        self._is_built = True
        with tf.compat.v1.variable_scope(self.name, values=[x]):
            with arg_scope(_arg_scope(is_training,
                                      self._weight_decay,
                                      self._batch_norm_decay,
                                      self._data_format)):
                return self._squeezenet(x, data_format=self._data_format)

    @staticmethod
    def _squeezenet(images, num_classes=200, data_format='NCHW'):
        net = conv2d(images, 96, [2, 2], scope='conv1')
        net = max_pool2d(net, [2, 2], scope='maxpool1')
        net = fire_module(net, 16, 64, scope='fire2', data_format=data_format)
        net = fire_module(net, 16, 64, scope='fire3', data_format=data_format)
        net = fire_module(net, 32, 128, scope='fire4', data_format=data_format)
        net = max_pool2d(net, [2, 2], scope='maxpool4')
        net = fire_module(net, 32, 128, scope='fire5', data_format=data_format)
        net = fire_module(net, 48, 192, scope='fire6', data_format=data_format)
        net = fire_module(net, 48, 192, scope='fire7', data_format=data_format)
        net = fire_module(net, 64, 256, scope='fire8', data_format=data_format)
        net = max_pool2d(net, [2, 2], scope='maxpool8')
        net = fire_module(net, 64, 256, scope='fire9', data_format=data_format)
        net = avg_pool2d(net, [7, 7], scope='avgpool10')
        net = conv2d(net, num_classes, [1, 1],
                     activation_fn=None,
                     normalizer_fn=None,
                     scope='conv10')

        squeeze_axes = [2, 3] if data_format == 'NCHW' else [1, 2]
        logits = tf.squeeze(net, squeeze_axes, name='logits')
        return logits


def _arg_scope(is_training, weight_decay, bn_decay, data_format):
    with arg_scope([conv2d],
                   weights_regularizer=l2_regularizer(weight_decay),
                   normalizer_fn=batch_norm,
                   normalizer_params={'is_training': is_training,
                                      'fused': True,
                                      'decay': bn_decay}):
        with arg_scope([conv2d, avg_pool2d, max_pool2d, batch_norm],
                       data_format=data_format) as sc:
                return sc


'''
Network in Network: https://arxiv.org/abs/1312.4400
See Section 3.2 for global average pooling
'''
