import tensorflow as tf

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import time

# Set up some global values here
content_path = 'images/Green_Sea_Turtle_grazing_seagrass.jpg'
style_path = 'images/The_Great_Wave_off_Kanagawa.jpg'


def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


# load content and style image
content_img = load_img(content_path)
style_img = load_img(style_path)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.squeeze(content_img, axis=0).astype('uint8'))
axs[0].set_title("Content Image")
axs[1].imshow(np.squeeze(style_img, axis=0).astype('uint8'))
axs[1].set_title("Style Image")
plt.show()


# load and process image
def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


# deprocess image
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


# feature extracting model
def get_model():
    """ Creates our model with access to intermediate layers.

    This function will load the VGG19 model and access the intermediate layers.
    These layers will then be used to create a new model that will take input image
    and return the outputs from these intermediate layers from the VGG model.

    Returns:
      returns a keras model that takes image inputs and outputs the style and
        content intermediate layers.
    """
    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2']

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1'
                    ]

    # Load our model. We load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # Get output layers corresponding to style and content layers
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build model
    return models.Model(vgg.input, model_outputs)


# define loss
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    """Expects two images of dimension h, w, c"""
    # height, width, num filters of each layer
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))  # / (4. * (channels ** 2) * (width * height) ** 2)


def get_feature_representations(model, image_path):
    # Load image
    image = load_and_process_img(image_path)

    # batch compute content and style features
    feats = model(image)

    # Get the style and content feature representations from our model
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    """This function will compute the loss total loss.

    Arguments:
      model: The model that will give us access to the intermediate layers
      loss_weights: The weights of each contribution of each loss function.
        (style weight, content weight, and total variation weight)
      init_image: Our initial base image. This image is what we are updating with
        our optimization process. We apply the gradients wrt the loss we are
        calculating to this image.
      gram_style_features: Precomputed gram matrices corresponding to the
        defined style layers of interest.
      content_features: Precomputed outputs from defined content layers of
        interest.

    Returns:
      returns the total loss, style loss, content loss, and total variational loss
    """
    style_weight, content_weight = loss_weights

    # Feed our init image through our model. This will give us the content and
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


# def compute_grads(cfg):
#     with tf.GradientTape() as tape:
#         all_loss = compute_loss(**cfg)
#     # Compute gradients wrt input image
#     total_loss = all_loss[0]
#     return tape.gradient(total_loss, cfg['init_image']), all_loss


def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):
    # We don't need to (or want to) train any layers of our model, so we set their
    # trainable to false.
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    init_image = tf.clip_by_value(init_image, min_vals, max_vals)

    # For displaying intermediate images
    iter_count = 1

    # Store our best result
    best_loss, best_img = float('inf'), None

    # Create a nice config
    loss_weights = (style_weight, content_weight)
    loss, style_loss, content_loss = compute_loss(model, loss_weights, init_image, gram_style_features, content_features)

    # Create our optimizer
    opt = tf.train.AdamOptimizer(learning_rate=1e1, beta1=0.99, beta2=0.999, epsilon=1e-8).minimize(loss)

    # For displaying
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations / (num_rows * num_cols)
    start_time = time.time()
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    check_interval = 10
    imgs = []

    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        print('Optimization started ... ')
        for i in range(num_iterations):

            init_image_v, loss_v, style_score_v, content_score_v = sess.run([init_image, loss, style_loss, content_loss])
            opt.run()


            if loss_v < best_loss:
                # Update best loss and best image from total loss.
                best_loss = loss_v
                best_img = deprocess_img(init_image_v)

            if i % check_interval == 0:
                # Use the .numpy() method to get the concrete numpy array
                plot_img = init_image_v
                plot_img = deprocess_img(plot_img)
                imgs.append(plot_img)
                print('Iteration: {}'.format(i))
                print('Total loss: {:.4e}, '
                      'style loss: {:.4e}, '
                      'content loss: {:.4e}, '
                      'time: {:.4f}s'.format(loss_v, style_score_v, content_score_v, time.time() - start_time))
                start_time = time.time()
        print('Total time: {:.4f}s'.format(time.time() - global_start))

    return best_img, best_loss, imgs


# best, best_loss, images = run_style_transfer(content_path, style_path, num_iterations=100)


with tf.Graph().as_default():
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    # Set loss weights
    content_weight = 1e3
    style_weight = 1e-2
    loss_weights = (style_weight, content_weight)

    # content loss
    content_layers_weights = {}
    content_layers_weights['relu4_2'] = content_weight_blend
    content_layers_weights['relu5_2'] = 1.0 - content_weight_blend

    content_loss = 0
    content_losses = []
    for content_layer in CONTENT_LAYERS:
        content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(
            net[content_layer] - content_features[content_layer]) /
                                                                                        content_features[
                                                                                            content_layer].size))
    content_loss += reduce(tf.add, content_losses)


    loss, style_loss, content_loss = compute_loss(model, loss_weights, init_image, gram_style_features, content_features)

    # optimizer setup
    train_step = tf.train.AdamOptimizer(learning_rate=1e5, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)

    # optimization
    best_loss = float('inf')
    best = None
    iterations = 100
    check_intervals = 10
    images = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Optimization started...')
        for i in range(iterations):
            train_step.run()

            loss_v = loss.eval()
            if loss_v < best_loss:
                best_loss = loss_v
                best = init_image.eval()
                images.append(best)
                print("Loss: %.4e" % loss_v)
