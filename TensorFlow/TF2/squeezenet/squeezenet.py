import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout, Concatenate, Activation
import argparse
import os
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = (len(tf.config.list_physical_devices('GPU')) > 0)

tf.random.set_seed(1234)

parser = argparse.ArgumentParser(description='Squeezenet model for TF2')

parser.add_argument('--mode', type=str, default='train', help='Can be "train" or "test"')
parser.add_argument('--checkpoint_dir', type=str, default='./', help='Directory to store checkpoints during training')
parser.add_argument('--restore_checkpoint', action='store_true', help='Use this flag if you want to resume training from a previous checkpoint')
parser.add_argument('--batch_size', type=int, default=512, help='Number of images per batch fed through network')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of passes through training data before stopping')
parser.add_argument("--log_device_placement", action="store_true", help="Print the operator device placement on the pre-optimized graph")
parser.add_argument('--device', type=str, default='CPU:0' if not gpu_available else 'GPU:0', help='Specify manually to use non-DML GPU device eg. CPU:0')
parser.add_argument("--inter_op_threads", default=0, type=int, help="Max number of threads for the runtime to use for kernel scheduling")
parser.add_argument('--cifar10', action='store_true', help='Train with CIFAR-10 dataset')
parser.add_argument('--tb_profile', action='store_true', help='Performance profiling using TensorBoard')

args = parser.parse_args()


class FireModule(tf.keras.layers.Layer):
    def __init__(self, squeeze, expand, name='', training=True):
        super(FireModule, self).__init__()

        self.squeeze = Conv2D(squeeze, (1,1), strides=1, activation='relu')
        self.expand_1 = Conv2D(expand, (1,1), strides=1, activation='relu')
        self.expand_3 = Conv2D(expand, (3,3), strides=1, padding='same', activation='relu')
        self.concat = Concatenate(axis=-1)
    
    @tf.function
    def call(self, inputs):
        out = self.squeeze(inputs)
        out_l = self.expand_1(out)
        out_r = self.expand_3(out)
        out = self.concat((out_l, out_r))
        return out

class SqueezeNet(Model):
    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()

        self.num_classes = num_classes
        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(96, (7,7), strides=(2,2), padding='same', activation='relu', name='conv_1'))
        self.model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), name='maxpool_2'))
        self.model.add(FireModule(squeeze=16, expand=64, name='fire_3'))
        self.model.add(FireModule(squeeze=16, expand=64, name='fire_4'))
        self.model.add(FireModule(squeeze=32, expand=128, name='fire_5'))
        self.model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), name='maxpool_6'))
        self.model.add(FireModule(squeeze=32, expand=128, name='fire_7'))
        self.model.add(FireModule(squeeze=48, expand=192, name='fire_8'))
        self.model.add(FireModule(squeeze=48, expand=192, name='fire_9'))
        self.model.add(FireModule(squeeze=64, expand=256, name='fire_10'))
        self.model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), name='maxpool_11'))
        self.model.add(FireModule(squeeze=64, expand=256, name='fire_12'))
        self.model.add(Dropout(0.5, name='dropout_13'))
        self.model.add(Conv2D(self.num_classes, (1,1), padding='valid', activation='relu', name='conv_14'))
        self.model.add(GlobalAveragePooling2D(name='avgpool_15'))
        self.model.add(Activation(tf.nn.softmax, name='softmax'))

    @tf.function
    def call(self, inputs):
        out = self.model(inputs)
        return out

# Modified SqueezeNet model for CIFAR-10 dataset
class SqueezeNet_CIFAR(Model):
    def __init__(self, num_classes=10):
        super(SqueezeNet_CIFAR, self).__init__()

        self.num_classes = num_classes
        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(96, (3,3), strides=(2,2), padding='same', activation='relu', name='conv_1'))
        self.model.add(MaxPool2D(pool_size=(2,2), name='maxpool_2'))
        self.model.add(FireModule(squeeze=16, expand=64, name='fire_3'))
        self.model.add(FireModule(squeeze=16, expand=64, name='fire_4'))
        self.model.add(FireModule(squeeze=32, expand=128, name='fire_5'))
        self.model.add(MaxPool2D(pool_size=(2,2), name='maxpool_6'))
        self.model.add(FireModule(squeeze=32, expand=128, name='fire_7'))
        self.model.add(Dropout(0.5, name='dropout_8'))
        self.model.add(Conv2D(self.num_classes, (1,1), padding='valid', activation='relu', name='conv_9'))
        self.model.add(GlobalAveragePooling2D(name='avgpool_10'))
        self.model.add(Activation(tf.nn.softmax, name='softmax'))

    @tf.function
    def call(self, inputs):
        out = self.model(inputs)
        return out

# Download and preprocess CIFAR-10 dataset
def get_cifar10_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def main():
    if args.log_device_placement:
        tf.debugging.set_log_device_placement(True)
    if args.inter_op_threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(args.inter_op_threads)
    if args.device[:3] != 'GPU':
        tf.config.set_visible_devices([], 'GPU')

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    train_dataset, test_dataset = get_cifar10_data()

    ckpt_dir = os.path.join(args.checkpoint_dir, "checkpoints/")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    try:
        with tf.device('/device:' + args.device):
            if args.cifar10:
                model = SqueezeNet_CIFAR()
            else:
                model = SqueezeNet()

            if args.restore_checkpoint or args.mode == 'test':
                model.load_weights(ckpt_dir).expect_partial()

            if args.mode == 'train':
                x_train, y_train = train_dataset
                cbs = []
                if args.tb_profile:
                    profile_dir = os.path.join(args.checkpoint_dir, "train/")
                    # Previously-existing profiler directory will be deleted
                    if os.path.exists(profile_dir):
                        shutil.rmtree(profile_dir)
                    os.makedirs(profile_dir)
                    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = profile_dir, histogram_freq=1, profile_batch='500,520')
                    cbs.append(tboard_callback)
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_dir,
                                                            save_weights_only=True,
                                                            verbose=1)
                cbs.append(cp_callback)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.04, epsilon=1.0), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                history = model.fit(x_train, y_train, epochs=num_epochs, shuffle=False, batch_size=batch_size, validation_data=test_dataset, callbacks=cbs)
                train_loss = history.history['loss']
                train_acc = history.history['accuracy']

            if args.mode == 'test':
                x_test, y_test = test_dataset
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.04, epsilon=1.0), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    main()