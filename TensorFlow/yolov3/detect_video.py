import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, get_class_colors
import os
from tensorflow.python.client import timeline
import numpy as np
import json

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_boolean('trace', False, 'trace each frame')
flags.DEFINE_boolean('headless', False, 'do not display frames (useful for tracing)')
flags.DEFINE_integer('max_frames', 0, 'max number of video frames to process (defaults to all)')
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
        trace_dir = os.path.join("traces", "predict_video")
        trace_basename = os.path.splitext(os.path.basename(FLAGS.video))[0]
        if not os.path.isdir(trace_dir):
            os.makedirs(trace_dir)
        graphs_dir = os.path.join("traces", "predict_video", "graphs")
        if not os.path.isdir(graphs_dir):
            os.makedirs(graphs_dir)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    class_colors = get_class_colors(FLAGS.colors, class_names)
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_index = 0

    while True:
        if FLAGS.max_frames and (frame_index > FLAGS.max_frames):
            break

        _, img = vid.read()

        if img is None:
            break
            # logging.warning("Empty Frame")
            # time.sleep(0.1)
            # continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = cv2.resize(img_in, dsize=(FLAGS.size, FLAGS.size))
        img_in = np.expand_dims(img_in, 0) / 255.0

        t1 = time.time()
        if FLAGS.eager:
            boxes, scores, classes, nums = yolo.predict(img_in)
        else:
            boxes, scores, classes, nums = sess.run(
                yolo.output, 
                feed_dict={yolo.input: img_in}, 
                options=run_options, 
                run_metadata=run_metadata)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        if FLAGS.trace:
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(os.path.join(trace_dir, f"{trace_basename}_{frame_index}.json"), 'w') as f:
                f.write(chrome_trace)
            # No need to dump graph partitions for every frame; they should be identical.
            if frame_index == 0:
                for i in range(len(run_metadata.partition_graphs)):
                    with open(os.path.join(graphs_dir, f"partition_{i}.pbtxt"), 'w') as f:
                        f.write(str(run_metadata.partition_graphs[i]))

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names, class_colors)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(img)
        if not FLAGS.headless:
            cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

        frame_index += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
