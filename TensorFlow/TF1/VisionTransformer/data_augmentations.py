import tensorflow.compat.v1 as tf
import numpy as np
tf.enable_eager_execution()

def flip_horizontal(img):
    img = tf.image.random_flip_left_right(img)
    return img

def flip_vertical(img):
    img = tf.image.random_flip_up_down(img)
    return img

def random_hue_saturation(img):
    img = tf.image.random_hue(img, 0.08)
    img = tf.image.random_saturation(img, 2, 5)

    return img

def random_brightness_contrast(img):
    
    #Adjusts contrast using a contrast_factor chosen randomly from the range [-0.25,8]
    img = tf.image.random_contrast(img, 0.25, 0.8)

    # Adjusts brightness using a delta chosen randomly from the range [-0.4,0.4]
    img = tf.image.random_brightness(img, 0.4)
    return img


def random_zoom_crop(img: tf.Tensor) -> tf.Tensor:
    zoom_levels = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(zoom_levels), 4))

    for i, scale in enumerate(zoom_levels):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(zoom_levels)), crop_size=(32, 32))
    return crops[tf.random_uniform(shape=[], minval=0, maxval=len(zoom_levels), dtype=tf.int32)]

