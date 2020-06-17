import os
import tensorflow as tf
import shutil
import argparse
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

def _parse_args():
    parser = argparse.ArgumentParser("save_squeezenet.py")
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='''Directory of the checkpoint to run inference on.''')

    return parser.parse_args()


def main():
    args = _parse_args()

    trained_checkpoint_prefix = tf.train.latest_checkpoint(args.model_dir)

    # Each model folder must be named '0', '1', ...
    export_dir = os.path.join(args.model_dir, 'models', '0')
    shutil.rmtree(export_dir, ignore_errors=True)

    with tf.compat.v1.Session(
            graph=tf.Graph(),
            config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        # Restore from checkpoint
        loader = tf.compat.v1.train.import_meta_graph(
                trained_checkpoint_prefix + '.meta')

        loader.restore(sess, trained_checkpoint_prefix)
        
        # Export checkpoint to SavedModel
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)

        images = sess.graph.get_tensor_by_name('inputs/split_images:0')
        is_training = sess.graph.get_tensor_by_name('inputs/is_training:0')
        predictions = sess.graph.get_tensor_by_name('predictions:0')

        signature = predict_signature_def(
                inputs={'images': images, 'is_training': is_training},
                outputs={'predictions': predictions})

        builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.SERVING],
                strip_default_attrs=True,
                signature_def_map={'predict': signature})

        builder.save()


if __name__ == '__main__':
    main()
