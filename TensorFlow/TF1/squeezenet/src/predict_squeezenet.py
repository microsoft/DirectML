import os
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

def _parse_args():
    parser = argparse.ArgumentParser("run_squeezenet.py")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory of the checkpoint to run inference on.")

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the 32x32 image to classify. Can be a folder or a single "
             "image.")

    parser.add_argument(
        '--data_format',
        default='NCHW',
        choices=['NCHW', 'NHWC'])

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help="How many images to predict in a single inference call. This "
             "parameter is useful when a large folder is given to --image to "
             "avoid running out of memory.")

    return parser.parse_args()

def main():
    with tf.compat.v1.Session(
            graph=tf.Graph(),
            config=tf.compat.v1.ConfigProto()) as sess:
        args = _parse_args()

        image_paths = []

        if os.path.isdir(args.image):
            for file_path in os.listdir(args.image):
                full_path = os.path.join(args.image, file_path)
                if os.path.isfile(full_path):
                    image_paths.append(full_path)
        else:
            image_paths.append(args.image)

        image_batches = []
        image_path_batches = []

        for i in range(0, len(image_paths), args.batch_size):
            image_path_batch = image_paths[i:i + args.batch_size]
            image_path_batches.append(image_path_batch)
            image_batch = []

            for image_path in image_path_batch:
                image = plt.imread(image_path)
                image = np.expand_dims(image, axis=0)
                image_batch.append(image)

            image_batch = np.concatenate(image_batch, axis=0)

            if args.data_format == "NCHW":
                image_batch = np.transpose(image_batch, [0, 3, 1, 2])

            image_batches.append(image_batch)

        model = tf.compat.v2.saved_model.load(
                os.path.join(args.model_dir, "models", "0"))
        labels_path = os.path.join(args.model_dir, "labels.txt")
        labels = np.array(open(labels_path).read().splitlines())
        labels = [label.split(":")[1] for label in labels]
        predict = model.signatures["predict"]

        for image_batch, image_paths in zip(image_batches, image_path_batches):
            image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)
            squeezenet = predict(images=image_batch,
                                 is_training=tf.constant(False))
            sess.run(tf.compat.v1.global_variables_initializer())
            predictions = sess.run(squeezenet)["predictions"]

            for image_path, prediction in zip(image_paths, predictions):
                label = labels[prediction]
                print(f"{image_path}: predicted {label}")

if __name__ == '__main__':
    main()
