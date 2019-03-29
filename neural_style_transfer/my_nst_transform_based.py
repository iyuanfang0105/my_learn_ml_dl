import os
import tensorflow as tf
import numpy as np
from scipy import misc

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class ImageUtils(object):
    def __init__(self):
        print("Initial Image Utils!!!")

    @staticmethod
    def save_img(out_path, img):
        img = np.clip(img, 0, 255).astype(np.uint8)
        misc.imsave(out_path, img)

    @staticmethod
    def scale_img(style_path, style_scale):
        scale = float(style_scale)
        o0, o1, o2 = misc.imread(style_path, mode='RGB').shape
        scale = float(style_scale)
        new_shape = (int(o0 * scale), int(o1 * scale), o2)
        style_target = ImageUtils.get_img(style_path, img_size=new_shape)
        return style_target

    @staticmethod
    def get_img(src, img_size=False):
        img = misc.imread(src, mode='RGB')  # misc.imresize(, (256, 256, 3))
        if not (len(img.shape) == 3 and img.shape[2] == 3):
            img = np.dstack((img, img, img))
        if img_size != False:
            img = misc.imresize(img, img_size)
        return img


class ImageTransformNet(object):
    def __init__(self):
        print("Initial Image Transform Network!!!")
        self.WEIGHTS_INIT_STDEV = .1

    def net(self, image):
        conv1 = self._conv_layer(image, 32, 9, 1)
        conv2 = self._conv_layer(conv1, 64, 3, 2)
        conv3 = self._conv_layer(conv2, 128, 3, 2)
        resid1 = self._residual_block(conv3, 3)
        resid2 = self._residual_block(resid1, 3)
        resid3 = self._residual_block(resid2, 3)
        resid4 = self._residual_block(resid3, 3)
        resid5 = self._residual_block(resid4, 3)
        conv_t1 = self._conv_tranpose_layer(resid5, 64, 3, 2)
        conv_t2 = self._conv_tranpose_layer(conv_t1, 32, 3, 2)
        conv_t3 = self._conv_layer(conv_t2, 3, 9, 1, relu=False)
        preds = tf.nn.tanh(conv_t3) * 150 + 255. / 2
        return preds

    def _conv_layer(self, net, num_filters, filter_size, strides, relu=True):
        weights_init = self._conv_init_vars(net, num_filters, filter_size)
        strides_shape = [1, strides, strides, 1]
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
        net = self._instance_norm(net)
        if relu:
            net = tf.nn.relu(net)

        return net

    def _conv_tranpose_layer(self, net, num_filters, filter_size, strides):
        weights_init = self._conv_init_vars(net, num_filters, filter_size, transpose=True)

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1, strides, strides, 1]

        net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
        net = self._instance_norm(net)
        return tf.nn.relu(net)

    def _residual_block(self, net, filter_size=3):
        tmp = self._conv_layer(net, 128, filter_size, 1)
        return net + self._conv_layer(tmp, 128, filter_size, 1, relu=False)

    def _instance_norm(self, net, train=True):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
        shift = tf.Variable(tf.zeros(var_shape))
        scale = tf.Variable(tf.ones(var_shape))
        epsilon = 1e-3
        normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
        return scale * normalized + shift

    def _conv_init_vars(self, net, out_channels, filter_size, transpose=False):
        _, rows, cols, in_channels = [i.value for i in net.get_shape()]
        if not transpose:
            weights_shape = [filter_size, filter_size, in_channels, out_channels]
        else:
            weights_shape = [filter_size, filter_size, out_channels, in_channels]

        weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=self.WEIGHTS_INIT_STDEV, seed=1),
                                   dtype=tf.float32)
        return weights_init


if __name__ == '__main__':
    device = '/cpu:0'
    # load test image
    test_image_path = 'images/Green_Sea_Turtle_grazing_seagrass.jpg'
    image_utils = ImageUtils()
    test_image = ImageUtils.get_img(test_image_path)

    # show test image
    plt.imshow(test_image)
    plt.show()

    # build image transform network
    image_transform_net = ImageTransformNet()

    # pretrained weights and graph
    pretrained_itn_ckpt_path = 'pretrained_model/wave.ckpt'

    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:
        img_placeholder = tf.placeholder(name='img_placeholder', shape=(1,) + test_image.shape, dtype=tf.float32)
        preds = image_transform_net.net(img_placeholder)

        saver = tf.train.Saver()
        saver.restore(sess, pretrained_itn_ckpt_path)

        pred_v = sess.run(preds, feed_dict={img_placeholder: np.expand_dims(test_image, axis=0)})

        print()

