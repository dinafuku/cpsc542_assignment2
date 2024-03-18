import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from preprocess import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.data.experimental import save as tf_save
from sklearn.metrics import accuracy_score, jaccard_score
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Dropout, UpSampling2D, BatchNormalization

start = time.time()

# get data directory respectively
parent_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(parent_dir, '..', 'data', 'images')
mask_dir = os.path.join(parent_dir, '..', 'data', 'masks')

# preprocess images/masks
images, masks = load_and_preprocess_data(image_dir, mask_dir, 1000)

# normalize pixel values between [0,1]
images = images / 255.0
masks = masks / 255.0

# Train test validation split (60-20-20)
X_train_val, X_test, y_train_val, y_test = train_test_split(images, masks, test_size=0.2, random_state=542)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=542)

# save data for analysis
np.savez('split_data.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

# batch size
batch_size = 32

# create training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)

# create validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size)

# create testing dataset
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size)

# serialize and save the datasets
train_dataset_dir = os.path.join(parent_dir, 'train_dataset')
val_dataset_dir = os.path.join(parent_dir, 'val_dataset')
test_dataset_dir = os.path.join(parent_dir, 'test_dataset')

tf_save(train_dataset, train_dataset_dir)
tf_save(val_dataset, val_dataset_dir)
tf_save(test_dataset, test_dataset_dir)

# modular decoder block for decoder part of UNET model
def decoder_block(first, x, skip, filters, dropout_rate=0.2):
    if first:
        x = UpSampling2D((4, 4))(x)
    else:
        x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, skip])
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return x

def unet_model():
    # ENCODER
    # utilizes transfer learning model mobilenetv2 for encoder
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # freeze weights of base_model
    for layer in base_model.layers:
        layer.trainable = False
    # get output of encoder/base layer
    encoder_output = base_model.output

    # DECODER
    x = Conv2D(512, 3, activation='relu', padding='same')(encoder_output)
    x = BatchNormalization()(x)
    x = decoder_block(True, x, base_model.get_layer('block_6_expand_relu').output, 256)
    x = decoder_block(False, x, base_model.get_layer('block_3_expand_relu').output, 128)
    x = decoder_block(False, x, base_model.get_layer('block_1_expand_relu').output, 64)

    # upsample to match output size with input size
    x = UpSampling2D((2, 2))(x)

    # output 1 channel mask and use sigmoid for binary classification
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    # create model
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

# create the U-Net model
unet = unet_model()

# compile the model with a learning rate, binary crossentropy (binary mask), and accuracy
unet.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# print model summary
unet.summary()

# configure early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# defining a leanring rate scheduler
def lr_schedule(epoch):
    initial_lr = 0.001
    decay_rate = 0.9
    decay_step = 10
    lr = initial_lr * (decay_rate ** (epoch // decay_step))
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# define the checkpoint directory
checkpoint_dir = os.path.join(parent_dir, '..', 'models', 'checkpoint')

# ensure directory exists
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# define the checkpoint filepath
checkpoint_filepath = os.path.join(checkpoint_dir, 'model_checkpoint.h5')

# define the checkpoint callback to save weights and training history
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# fit UNET model
history = unet.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[early_stopping, lr_scheduler, checkpoint_callback])

# save the training history to a numpy file
np.save(os.path.join(checkpoint_dir, 'training_history.npy'), history.history)

# save the trained model
model_save_path = "unet_model.h5"
unet.save(model_save_path)
print("Model saved at:", model_save_path)