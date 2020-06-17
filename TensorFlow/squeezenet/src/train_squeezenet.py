import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys

import tensorflow as tf

from models.research.slim.deployment import model_deploy

from squeezenet import inputs
from squeezenet import networks
from squeezenet import arg_parsing
from squeezenet import metrics

from tensorflow.python.client import timeline
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.enable_resource_variables()

def _run(args):
    network = networks.catalogue[args.network](args)

    deploy_config = _configure_deployment(args.num_gpus, args.clone_on_cpu)
    sess = tf.compat.v1.Session(config=_configure_session())

    with tf.device(deploy_config.variables_device()):
        global_step = tf.compat.v1.train.create_global_step()

    with tf.device(deploy_config.optimizer_device()):
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=args.learning_rate
        )

    '''Inputs'''
    with tf.device(deploy_config.inputs_device()), tf.name_scope('inputs'):
        pipeline = inputs.Pipeline(args, sess)
        examples, labels = pipeline.data
        images = examples['image']

        image_splits = tf.split(
            value=images,
            num_or_size_splits=deploy_config.num_clones,
            name='split_images'
        )


        label_splits = tf.split(
            value=labels,
            num_or_size_splits=deploy_config.num_clones,
            name='split_labels'
        )

    '''Model Creation'''
    model_dp = model_deploy.deploy(
        config=deploy_config,
        model_fn=_clone_fn,
        optimizer=optimizer,
        kwargs={
            'images': image_splits,
            'labels': label_splits,
            'index_iter': iter(range(deploy_config.num_clones)),
            'network': network,
            'is_training': pipeline.is_training
        }
    )

    '''Metrics'''
    train_metrics = metrics.Metrics(
        labels=labels,
        clone_predictions=[clone.outputs['predictions']
                           for clone in model_dp.clones],
        device=deploy_config.variables_device(),
        name='training'
    )
    validation_metrics = metrics.Metrics(
        labels=labels,
        clone_predictions=[clone.outputs['predictions']
                           for clone in model_dp.clones],
        device=deploy_config.variables_device(),
        name='validation',
        padded_data=True
    )
    validation_init_op = tf.group(
        pipeline.validation_iterator.initializer,
        validation_metrics.reset_op
    )
    train_op = tf.group(
        model_dp.train_op,
        train_metrics.update_op
    )

    '''Summaries'''
    with tf.device(deploy_config.variables_device()):
        train_writer = tf.compat.v1.summary.FileWriter(args.model_dir, sess.graph)
        eval_dir = os.path.join(args.model_dir, 'eval')
        eval_writer = tf.compat.v1.summary.FileWriter(eval_dir, sess.graph)
        tf.compat.v1.summary.scalar('accuracy', train_metrics.accuracy)
        tf.compat.v1.summary.scalar('loss', model_dp.total_loss)
        all_summaries = tf.compat.v1.summary.merge_all()

    if args.keep_last_n_checkpoints:
        '''Model Checkpoints'''
        saver = tf.compat.v1.train.Saver(max_to_keep=args.keep_last_n_checkpoints)
        save_path = os.path.join(args.model_dir, 'model.ckpt')

    '''Model Initialization'''
    last_checkpoint = tf.train.latest_checkpoint(args.model_dir)

    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                       tf.compat.v1.local_variables_initializer())
    sess.run(init_op)

    if args.keep_last_n_checkpoints and last_checkpoint:
        saver.restore(sess, last_checkpoint)

    starting_step = sess.run(global_step)

    if args.trace:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        options = None
        run_metadata = None

    '''Main Loop'''
    for train_step in range(starting_step, args.max_train_steps):
        sess.run(train_op, feed_dict=pipeline.training_data, options=options, run_metadata=run_metadata)

        '''Summary Hook'''
        if args.summary_interval > 0 and train_step % args.summary_interval == 0:
            results = sess.run(
                fetches={'accuracy': train_metrics.accuracy,
                         'summary': all_summaries},
                feed_dict=pipeline.training_data
            )
            train_writer.add_summary(results['summary'], train_step)
            print('Train Step {:<5}:  {:>.4}'
                  .format(train_step, results['accuracy']))

        if args.keep_last_n_checkpoints:
            '''Checkpoint Hooks'''
            if args.checkpoint_interval > 0 and train_step % args.checkpoint_interval == 0:
                saver.save(sess, save_path, global_step)

        sess.run(train_metrics.reset_op)

        '''Eval Hook'''
        if args.validation_interval > 0 and train_step % args.validation_interval == 0:
            while True:
                try:
                    sess.run(
                        fetches=validation_metrics.update_op,
                        feed_dict=pipeline.validation_data
                    )
                except tf.errors.OutOfRangeError:
                    break
            results = sess.run({'accuracy': validation_metrics.accuracy})

            print('Evaluation Step {:<5}:  {:>.4}'
                  .format(train_step, results['accuracy']))

            summary = tf.compat.v1.Summary(value=[
                    tf.compat.v1.Summary.Value(
                            tag='accuracy', simple_value=results['accuracy']),
            ])
            eval_writer.add_summary(summary, train_step)
            sess.run(validation_init_op)  # Reinitialize dataset and metrics

        if args.trace:
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(os.path.join(args.model_dir, f'cifar_trace_{train_step}.json'), 'w') as f:
                f.write(chrome_trace)


def _clone_fn(images,
              labels,
              index_iter,
              network,
              is_training):
    clone_index = next(index_iter)
    images = images[clone_index]
    labels = labels[clone_index]

    unscaled_logits = network.build(images, is_training)
    tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels,
                                                     logits=unscaled_logits)
    predictions = tf.argmax(unscaled_logits, 1, name='predictions')
    return {
        'predictions': predictions,
        'images': images,
    }


def _configure_deployment(num_gpus, clone_on_cpu):
    return model_deploy.DeploymentConfig(num_clones=num_gpus,
                                         clone_on_cpu=clone_on_cpu)


def _configure_session():
    gpu_config = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=.8)
    return tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_config)


def run(args=None):
    args = arg_parsing.ArgParser().parse_args(args)
    with tf.Graph().as_default():
        _run(args)


if __name__ == '__main__':
    run()
