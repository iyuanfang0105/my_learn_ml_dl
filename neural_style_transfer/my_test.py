import tensorflow as tf

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras.optimizers import Adam

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import time


class FeatModel(object):
    def __init__(self, name='VGG19', content_feat_layers=[], style_feat_layers=[]):
        self.name = name
        self.content_feat_layers = content_feat_layers
        self.style_feat_layers = style_feat_layers
        self.model = None

    def build_model(self):
        if self.name == 'VGG19':
            # Load our model. We load pretrained VGG, trained on imagenet data
            vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
            vgg.trainable = False
            # Get output layers corresponding to style and content layers
            style_outputs = [vgg.get_layer(name).output for name in self.style_feat_layers]
            content_outputs = [vgg.get_layer(name).output for name in self.content_feat_layers]
            model_outputs = style_outputs + content_outputs
            # Build model
        else:
            print("Error: Not Support this model! Please Check!")

        self.model = models.Model(vgg.input, model_outputs)
        print(self.model.summary())

    def get_image_features(self, image):
        # img, image_ext = self.load_and_process_img(img_path)
        features = {}
        for layer, feat in zip(self.style_feat_layers + self.content_feat_layers, self.model(image)):
            features[layer] = feat
        return features

    def get_tensor_features(self, image_tensor):
        # img, image_ext = self.load_and_process_img(img_path)
        features = {}
        for layer, feat in zip(self.style_feat_layers + self.content_feat_layers, self.model(image_tensor)):
            features[layer] = feat
        return features

    @staticmethod
    def load_and_process_img(img_path):
        max_dim = 512
        img = Image.open(img_path)
        long = max(img.size)
        scale = max_dim / long
        img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)

        # We need to broadcast the image array such that it has a batch dimension
        img_ext = np.expand_dims(kp_image.img_to_array(img), axis=0)
        img_ext = tf.keras.applications.vgg19.preprocess_input(img_ext)
        return img, img_ext


class NeuralStyleTransfer(object):
    def __init__(self, content_feat_layers=[], style_feat_layers=[], content_weight=1e3, style_weight=1e-2):
        self.content_feat_layers = content_feat_layers
        self.style_feat_layers = style_feat_layers
        self.content_weight = content_weight
        self.style_weight = style_weight

    def content_loss(self, content_image_feats, generate_image_feats):
        content_layer_losses = []
        layer_weight = 1.0 / len(self.content_feat_layers)
        for layer in self.content_feat_layers:
            content_layer_losses.append(
                layer_weight * tf.reduce_mean(tf.square(generate_image_feats[layer] - content_image_feats[layer])))
        loss = tf.reduce_sum(content_layer_losses)
        return loss

    def style_loss(self, style_image_feats, generate_image_feats):
        style_layers_losses = []
        layer_weight = 1.0 / len(self.style_feat_layers)
        for layer in self.style_feat_layers:
            style_layers_losses.append(
                layer_weight * tf.reduce_mean(tf.square(self.gram_matrix(generate_image_feats[layer]) - self.gram_matrix(style_image_feats[layer]))))
        loss = tf.reduce_sum(style_layers_losses)
        return loss

    @staticmethod
    def gram_matrix(input_tensor):
        # We make the image channels first
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)


# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

# build model
feat_model = FeatModel(name='VGG19', content_feat_layers=content_layers, style_feat_layers=style_layers)
feat_model.build_model()

# load content and style images
content_image_path = 'images/Green_Sea_Turtle_grazing_seagrass.jpg'
style_image_path = 'images/The_Great_Wave_off_Kanagawa.jpg'
content_image, content_image_ext = feat_model.load_and_process_img(content_image_path)
style_image, style_image_ext = feat_model.load_and_process_img(style_image_path)

# extract features
content_image_feats = {}
style_image_feats = feat_model.get_image_features(style_image_ext)

g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    for k, v in feat_model.get_image_features(content_image_ext).items():
        content_image_feats[k] = sess.run(v)


# show images
plt.subplot(121)
plt.imshow(content_image)
plt.subplot(122)
plt.imshow(style_image)
plt.show()


# initial generated image
generated_image = tf.Variable(content_image_ext, dtype=tf.float32)

# extract features of generated image
generated_image_feats = feat_model.get_tensor_features(generated_image)

neural_style_transfer = NeuralStyleTransfer(content_feat_layers=content_layers, style_feat_layers=style_layers,
                                            content_weight=1e3, style_weight=1e-2)
ct_loss = neural_style_transfer.content_loss(content_image_feats, generated_image_feats)
st_loss = neural_style_transfer.style_loss(style_image_feats, generated_image_feats)

loss = ct_loss + st_loss

# optimizer setup
train_step = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        ct_loss_v, st_loss_v, loss_v = sess.run([ct_loss, st_loss, loss])
        print("content loss: %.5f, style loss: %.5f, total loss: %.5f" % (ct_loss_v, st_loss_v, loss_v))
        train_step.run()
