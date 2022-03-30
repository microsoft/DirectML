import tensorflow as tf


class Metrics(object):
    def __init__(self, labels, clone_predictions, device, name, padded_data=False):
        self.labels = labels
        self.clone_predictions = clone_predictions
        self.device = device
        self.name = name
        self.padded_data = padded_data
        self.accuracy = None
        self.update_op = None
        self.reset_op = None
        self._generate_metrics()

    def _generate_metrics(self):
        with tf.compat.v1.variable_scope('metrics'), tf.device(self.device):
            with tf.compat.v1.variable_scope(self.name):
                predictions = tf.concat(
                    values=self.clone_predictions,
                    axis=0
                )

                if self.padded_data:
                    not_padded = tf.not_equal(self.labels, -1)
                    self.labels = tf.boolean_mask(self.labels, not_padded)
                    predictions = tf.boolean_mask(predictions, not_padded)

                self.accuracy, self.update_op = tf.compat.v1.metrics.accuracy(
                    labels=self.labels,
                    predictions=predictions
                )
                accuracy_vars = tf.contrib.framework.get_local_variables(
                    scope='metrics/{}/accuracy'.format(self.name)
                )
                self.reset_op = tf.compat.v1.variables_initializer(
                    var_list=accuracy_vars
                )
