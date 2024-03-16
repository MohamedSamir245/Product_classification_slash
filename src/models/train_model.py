import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator

from pathlib import Path

import seaborn as sns

sns.set_style('darkgrid')




from functions import create_tensorboard_callback
BATCH_SIZE = 32
IMAGE_SIZE = (224,224)

train_root = Path('./data/external/train')
test_root = Path('./data/external/test')
valid_root = Path('./data/external/valid')


# Rescale
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# data transfer from directories to batches
train_data = train_datagen.flow_from_directory(directory=train_root,
                                               batch_size=BATCH_SIZE,
                                               target_size=IMAGE_SIZE,
                                               class_mode="categorical")

test_data = test_datagen.flow_from_directory(directory=test_root,
                                             batch_size=BATCH_SIZE,
                                             target_size=IMAGE_SIZE,
                                             class_mode="categorical")

val_data = valid_datagen.flow_from_directory(directory=valid_root,
                                             batch_size=BATCH_SIZE,
                                             target_size=IMAGE_SIZE,
                                             class_mode="categorical")

# Create checkpoint callback
checkpoint_path = "./models/checkpoints/product_classification_model_checkpoint"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True,
                                                         monitor="val_accuracy",
                                                         save_best_only=True)

# Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 4 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",  # watch the val loss metric
                                                  patience=5,
                                                  restore_best_weights=True)  # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
    tf.keras.layers.RandomRotation(factor=0.05, seed=42),
    tf.keras.layers.RandomZoom(0.05, seed=42),
])

base_model = tf.keras.applications.InceptionV3(include_top=False,)

# 2. Freeze the base model
base_model.trainable = False
model_0 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3), name="input-layer"),
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling_layer"),
    tf.keras.layers.Dense(3, activation="softmax", name="output-layer")
])


# 9. Compile the model
model_0.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                metrics=["accuracy"])

history = model_0.fit(train_data,
                      epochs=10,
                      validation_data=val_data,
                    callbacks=[
                          early_stopping,
                          create_tensorboard_callback("./logs/training_logs",
                                                      "product_classification"),
                          checkpoint_callback,
                          reduce_lr
                      ])

model_0.save("./models/product_classification_model.h5")


base_model.trainable = True

# Un-freeze last 22 layers
for layer in base_model.layers[:-22]:
  layer.trainable = False

# Reduce learning rate (it's better to reduce lr with factor 10 before fine tuning)
# Recompile (we have to compile model every time there is a change)
model_0.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                metrics=["accuracy"])

history_tuned = model_0.fit(train_data,
                            epochs=10,
                            # steps_per_epoch=len(train_data),
                            validation_data=val_data,
                            # validation_steps=int(0.25*len(val_data)), 
                            callbacks=[
                                early_stopping,
                                create_tensorboard_callback("./logs/training_logs",
                                                            "tuned_product_classification"),
                                checkpoint_callback,
                                reduce_lr
                            ])

model_0.save( "./models/product_classification_model_tuned.h5")