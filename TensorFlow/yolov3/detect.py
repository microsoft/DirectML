import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs, get_class_colors
import os
from tensorflow.python.client import timeline
import numpy as np

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', None, 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_boolean('trace', False, 'produce a chrome trace')
flags.DEFINE_boolean('headless', False, 'do not display image (useful for tracing)')
flags.DEFINE_string('colors', './data/colors.json', 'path to class colors file')
flags.DEFINE_boolean('eager', False, 'enables eager execution (unless tracing)')

def main(_argv):
    if FLAGS.eager and not FLAGS.trace:
        # The upstream model is written for TF2, which enables eager execution by default.
        # Leave eager execution disabled when tracing, since TF1.15 doesn't appear to
        # support the same level of profiling detail with eager mode enabled.
        tf.compat.v1.enable_eager_execution()
    else:
        sess = tf.keras.backend.get_session()
        run_options = None
        run_metadata = None

    if FLAGS.trace:
        run_options = tf.compat.v1.RunOptions(
            output_partition_graphs=True, 
            trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
        trace_dir = os.path.join("traces", "predict_image")
        trace_basename = os.path.splitext(os.path.basename(FLAGS.image))[0]
        if not os.path.isdir(trace_dir):
            os.makedirs(trace_dir)
        graphs_dir = os.path.join("traces", "predict_image", "graphs")
        if not os.path.isdir(graphs_dir):
            os.makedirs(graphs_dir)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    class_colors = get_class_colors(FLAGS.colors, class_names, True)
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img, _label = next(iter(dataset.take(1)))
    else:
        img = cv2.imread(FLAGS.image).astype(np.float32) / 255.0

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_in = cv2.resize(img_in, dsize=(FLAGS.size, FLAGS.size))
    img_in = np.expand_dims(img_in, 0)

    t1 = time.time()
    if FLAGS.eager:
        boxes, scores, classes, nums = yolo(img_in)
    else:
        boxes, scores, classes, nums = sess.run(
            yolo.output, 
            feed_dict={yolo.input: img_in}, 
            options=run_options, 
            run_metadata=run_metadata)

    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    if FLAGS.trace:
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open(os.path.join(trace_dir, f"{trace_basename}.json"), 'w') as f:
            f.write(chrome_trace)
        for i in range(len(run_metadata.partition_graphs)):
            with open(os.path.join(graphs_dir, f"partition_{i}.pbtxt"), 'w') as f:
                f.write(str(run_metadata.partition_graphs[i]))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = draw_outputs(img, (boxes, scores, classes, nums), class_names, class_colors)

    if FLAGS.output:
        cv2.imwrite(FLAGS.output, img)
        logging.info('output saved to: {}'.format(FLAGS.output))

    if not FLAGS.headless:
        cv2.imshow('output', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
