import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image


class ImageUtils(object):
    def __init__(self):
        print("Initial utils for images")

    @staticmethod
    def load_image(img_path):
        max_dim = 512
        img = Image.open(img_path)
        long = max(img.size)
        scale = max_dim / long
        img = np.asarray(img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS),
                         dtype='uint8')
        return img

    @staticmethod
    def nst_unprocess_image(ori_im, image):
        original_image = np.clip(ori_im, 0, 255)
        styled_image = np.clip(image, 0, 255)

        # Luminosity transfer steps:
        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
        # 2. Convert stylized grayscale into YUV (YCbCr)
        # 3. Convert original image into YUV (YCbCr)
        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
        # 5. Convert recombined image from YUV back to RGB

        # 1
        styled_grayscale = ImageUtils.rgb2gray(styled_image)
        styled_grayscale_rgb = ImageUtils.gray2rgb(styled_grayscale)

        # 2
        styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

        # 3
        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

        # 4
        w, h, _ = original_image.shape
        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
        combined_yuv[..., 1] = original_yuv[..., 1]
        combined_yuv[..., 2] = original_yuv[..., 2]

        # 5
        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))
        return img_out

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    @staticmethod
    def gray2rgb(gray):
        w, h = gray.shape
        rgb = np.empty((w, h, 3), dtype=np.float32)
        rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
        return rgb


class FeatModel(object):
    def __init__(self, name='VGG19', weights_path=''):
        self.name = name
        self.VGG19_LAYERS = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )
        self.CONTENT_LAYERS = ['relu5_2']
        self.STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        self.load_net(weights_path)

    def load_net(self, data_path):
        data = scipy.io.loadmat(data_path)
        if 'normalization' in data:
            # old format, for data where
            # MD5(imagenet-vgg-verydeep-19.mat) = 8ee3263992981a1d26e73b3ca028a123
            mean_pixel = np.mean(data['normalization'][0][0][0], axis=(0, 1))
        else:
            # new format, for data where
            # MD5(imagenet-vgg-verydeep-19.mat) = 106118b7cf60435e6d8e04f6a6dc3657
            self.mean_pixel = data['meta']['normalization'][0][0][0][0][2][0][0]
        self.weights = data['layers'][0]

    def build_model(self, input_image, pooling):
        net = {}
        current = input_image
        for i, name in enumerate(self.VGG19_LAYERS):
            kind = name[:4]
            if kind == 'conv':
                if isinstance(self.weights[i][0][0][0][0], np.ndarray):
                    # old format
                    kernels, bias = self.weights[i][0][0][0][0]
                else:
                    # new format
                    kernels, bias = self.weights[i][0][0][2][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                current = self._conv_layer(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current)
            elif kind == 'pool':
                current = self._pool_layer(current, pooling)
            net[name] = current

        assert len(net) == len(self.VGG19_LAYERS)
        return net

    def _conv_layer(self, input, weights, bias):
        conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                            padding='SAME')
        return tf.nn.bias_add(conv, bias)

    def _pool_layer(self, input, pooling):
        if pooling == 'avg':
            return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                                  padding='SAME')
        else:
            return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                                  padding='SAME')

    def preprocess(self, image):
        return image - self.mean_pixel

    def unprocess(self, image):
        return image + self.mean_pixel


if __name__ == '__main__':
    # initial utils for image
    utils = ImageUtils()

    # load content and style images
    content_image_path = 'images/Green_Sea_Turtle_grazing_seagrass.jpg'
    style_image_path = 'images/The_Great_Wave_off_Kanagawa.jpg'
    content_image = utils.load_image(content_image_path)
    style_image = utils.load_image(style_image_path)

    # show images
    plt.subplot(121)
    plt.imshow(content_image)
    plt.subplot(122)
    plt.imshow(style_image)
    plt.show()

    # initial features extraction model
    vgg19_weights_path = 'pretrained_model/imagenet-vgg-verydeep-19.mat'
    feat_model = FeatModel(weights_path=vgg19_weights_path)

    # compute content features in feedforward mode
    content_features = {}
    g1 = tf.Graph()
    with g1.as_default(), tf.Session() as sess:
        image = tf.placeholder('float', shape=(1,) + content_image.shape)
        net = feat_model.build_model(image, pooling='max')
        content_pre = np.array([feat_model.preprocess(content_image)])

        for layer in feat_model.CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # compute style features in feedforward mode
    style_features = {}
    g2 = tf.Graph()
    with g2.as_default(), tf.Session() as sess:
        image = tf.placeholder('float', shape=(1,) + style_image.shape)
        net = feat_model.build_model(image, pooling='max')
        style_pre = np.array([feat_model.preprocess(style_image)])
        for layer in feat_model.STYLE_LAYERS:
            features = net[layer].eval(feed_dict={image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        image = tf.Variable(np.array([feat_model.preprocess(content_image)]).astype('float32'))
        net = feat_model.build_model(image, pooling='max')

        # content loss
        content_loss = 0
        layer_weights = 1.0 / len(feat_model.CONTENT_LAYERS)
        for layer in feat_model.CONTENT_LAYERS:
            content_loss += layer_weights * tf.reduce_mean(tf.square(net[layer] - content_features[layer]))

        # style loss
        style_loss = 0
        layer_weights = 1.0 / len(feat_model.STYLE_LAYERS)
        for layer in feat_model.STYLE_LAYERS:
            feat_tmp = net[layer]
            _, height, width, number = map(lambda i: i.value, feat_tmp.get_shape())
            size = height * width * number
            feat_tmp = tf.reshape(feat_tmp, (-1, number))
            feat_tmp_gram = tf.matmul(tf.transpose(feat_tmp), feat_tmp) / size
            style_loss += layer_weights * tf.reduce_mean(tf.square(feat_tmp_gram - style_features[layer]))

        # total loss
        content_weight = 1
        style_weight = 1
        loss = content_weight * content_loss + style_weight * style_loss

        # optimizer setup
        LEARNING_RATE = 1e1
        BETA1 = 0.9
        BETA2 = 0.999
        EPSILON = 1e-08
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2, epsilon=EPSILON).minimize(loss)

        # optimization
        best_loss = float('inf')
        best = None
        iterations = 1000
        check_interval = 10
        styled_imgs = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('Optimization started...')

            for i in range(iterations):
                train_step.run()
                if i % check_interval == 0:
                    loss_v, ct_loss_v, st_loss_v = sess.run([loss, content_loss, style_loss])
                    if loss_v < best_loss:
                        best_loss = loss_v
                        best = image.eval()
                        styled_imgs.append(ImageUtils.nst_unprocess_image(content_image, feat_model.unprocess(best[0])))
                        print("Total loss: %.5f, Content loss: %.5f, Style loss: %.5f" % (loss_v, ct_loss_v, st_loss_v))

