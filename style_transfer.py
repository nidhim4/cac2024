import os
import tensorflow as tf
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image
import tensorflow_hub as hub


def style_transfer(image_path, style_path, output_path, blend_factor=0.3
                   ):
    mpl.rcParams['figure.figsize'] = (12, 12)
    mpl.rcParams['axes.grid'] = False

    def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def load_img(path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def imshow(image, title=None):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)

        plt.imshow(image)
        if title:
            plt.title(title)

    content_image = load_img(image_path)
    style_image = load_img(style_path)

    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    blended_image = (1 - blend_factor) * content_image + blend_factor * stylized_image
    blended_image = tf.clip_by_value(blended_image, 0, 1)  # Ensure values are between 0 and 1

    output_image = tensor_to_image(blended_image)

    output_image.save(output_path)

    return output_path