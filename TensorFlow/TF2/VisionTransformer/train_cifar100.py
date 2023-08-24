import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model import VisionTransformer
from data_augmentations import random_hue_saturation, random_brightness_contrast, flip_horizontal, flip_vertical, random_zoom_crop

AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == "__main__":
    
    #You can uncomment the below line to view DirectML Device Placement logs for the model's operators
    # tf.debugging.set_log_device_placement(True) 

    IMAGE_SIZE= 32
    NUMBER_OF_CLASSES= 100
    PATCH_SIZE= 4
    PATCH_STRIDE=4
    NUMBER_OF_LAYERS=8
    EMBEDDING_DIM=64
    NUM_HEADS= 8
    MLP_HIDDEN_DIM= 256
    LEARNING_RATE= 0.001 
    BATCH_SIZE= 128
    EPOCHS= 100
    PATIENCE= 10 #Patience controls the number of epochs with no increase in validation accuracy the Learning Rate Scheduler will wait before reducing Learning Rate

    (X_train, y_train) , (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.15, shuffle=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_validate, y_validate))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    def cast_to_float(x,y):
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    train_dataset = train_dataset.map(cast_to_float)
    validation_dataset = validation_dataset.map(cast_to_float)


    def get_train_dataset(train_dataset):

        train_dataset= train_dataset.shuffle(10000, reshuffle_each_iteration= True)
        augmentations = [random_hue_saturation, random_zoom_crop, random_brightness_contrast, flip_horizontal,flip_horizontal, flip_vertical]
        for aug in augmentations:
            train_dataset = train_dataset.map(lambda x, y: (tf.cond(tf.random.uniform([], 0, 1) > 0.86, lambda: aug(x), lambda: x), y), num_parallel_calls=AUTOTUNE)
           
        train_dataset= train_dataset.cache()
        train_dataset=train_dataset.batch(BATCH_SIZE)
        train_dataset=train_dataset.prefetch(AUTOTUNE)
        
        return train_dataset

    validation_dataset = (
        validation_dataset
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    model = VisionTransformer(
        image_size= IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        num_layers=NUMBER_OF_LAYERS,
        num_classes=NUMBER_OF_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="Top-1-accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="Top-3-accuracy"),
        ],
    )
    
file_path= './saved_models/Model_Cifar100'
checkpoint = ModelCheckpoint(file_path, monitor='val_Top-1-accuracy', verbose=1, save_best_only=True, mode='max')
reduce_on_plateau = ReduceLROnPlateau(monitor="val_Top-1-accuracy", mode="max", factor=0.5, patience=PATIENCE, verbose=1,min_lr=0.00002)
callbacks_list = [checkpoint, reduce_on_plateau]


model.fit(
    get_train_dataset(train_dataset),
    validation_data=validation_dataset,
    validation_steps=min(10, len(validation_dataset)),
    epochs=EPOCHS,
    callbacks=callbacks_list,
)

#Compute Metrics on Test Set
test_metrics= model.evaluate(X_test, y_test, 128)
print(test_metrics)
