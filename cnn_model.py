import os
import numpy as np
import tensorflow as tf
from preprocess import load_and_preprocess_data
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, jaccard_score

# path to data
images_folder = 'data/images'
masks_folder = 'data/masks'

# preprocess images/masks
images, masks = load_and_preprocess_data(images_folder, masks_folder, 200)

# normalize pixel values between [0,1]
images = images / 255.0
masks = masks / 255.0

# Train test validation split (60-20-20)
X_train_val, X_test, y_train_val, y_test = train_test_split(images, masks, test_size=0.2, random_state=542)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=542)

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

# modular decoder block for decoder part of UNET model
def decoder_block(first, x, skip, filters):
    if first:
        x = UpSampling2D((4, 4))(x)
    else:
        x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, skip])
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
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

# fit UNET model
history = unet.fit(train_dataset, epochs=25, validation_data=val_dataset)

# plot training history
plt.figure(figsize=(12, 5))

# plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()

# create the "predicted_masks" folder if it does not exist
unet_results = "unet_results"
if not os.path.exists(unet_results):
    os.makedirs(unet_results)

# save figures
plt.savefig(os.path.join(unet_results, "training_history.png"))

# show plots
plt.show()

# evaluate the model on train, validation, and test datasets
train_loss, train_accuracy = unet.evaluate(train_dataset)
val_loss, val_accuracy = unet.evaluate(val_dataset)
test_loss, test_accuracy = unet.evaluate(test_dataset)

# print out train,validation,test performance
print("Train Accuracy:", train_accuracy)
print("Validation Accuracy:", val_accuracy)
print("Test Accuracy:", test_accuracy)
print("Train Loss:", train_loss)
print("Validation Loss:", val_loss)
print("Test Loss:", test_loss)

# save figures of original image + mask + predicted mask
def plot_and_save_images(images, true_masks, predicted_masks, index):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(images[index])
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(true_masks[index], cmap="gray")
    axes[1].set_title("True Mask")
    axes[1].axis("off")
    
    axes[2].imshow(predicted_masks[index], cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(unet_results, f"predicted_mask_{index}.png"))
    plt.close()

# get images and true masks from test dataset
for i, (images, true_masks) in enumerate(test_dataset):
    # predict mask
    predicted_masks = unet.predict(images)
    
    # plot and save images + true masks + predicted masks for some examples
    for j in range(min(len(images), 5)):
        plot_and_save_images(images, true_masks, predicted_masks, j)