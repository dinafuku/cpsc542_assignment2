import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from tf_explain.core.grad_cam import GradCAM
from tensorflow.keras.models import load_model
from tensorflow.data.experimental import load as tf_load

start = time.time()

# get data directory respectively
parent_dir = os.path.dirname(os.path.abspath(__file__))
saved_model_dir = os.path.join(parent_dir, '..', 'models', 'unet_model.h5')

# Load the saved model
unet = tf.keras.models.load_model(saved_model_dir)

# Load the data
parent_dir = os.path.dirname(os.path.abspath(__file__))
saved_data = os.path.join(parent_dir, '..', 'models', 'split_data.npz')
loaded_data = np.load(saved_data)

# Extract the data
X_train = loaded_data['X_train']
y_train = loaded_data['y_train']
X_val = loaded_data['X_val']
y_val = loaded_data['y_val']
X_test = loaded_data['X_test']
y_test = loaded_data['y_test']

# get dataset paths
train_dataset_dir = os.path.join(parent_dir, '..', 'models', 'train_dataset')
val_dataset_dir = os.path.join(parent_dir, '..', 'models', 'val_dataset')
test_dataset_dir = os.path.join(parent_dir, '..', 'models', 'test_dataset')

# load saved datasets
train_dataset = tf_load(train_dataset_dir)
val_dataset = tf_load(val_dataset_dir)
test_dataset = tf_load(test_dataset_dir)

checkpoint_dir = os.path.join(parent_dir, '..', 'models', 'checkpoint')
training_history_path = os.path.join(checkpoint_dir, 'training_history.npy')
training_history = np.load(training_history_path, allow_pickle=True).item()

# plot training history
plt.figure(figsize=(12, 5))

# plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(training_history['accuracy'])
plt.plot(training_history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(training_history['loss'])
plt.plot(training_history['val_loss'])
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

# compute IOU score using jaccard score for training dataset
train_scores = []
for images, true_masks in train_dataset:
    predicted_masks = unet.predict(images)
    for true_mask, predicted_mask in zip(true_masks, predicted_masks):
        # remove single dimension, convert to bool
        true_mask = true_mask.numpy().squeeze().astype(bool)  
        # threshold the mask using 0.5 as convert to boolean
        predicted_mask = (predicted_mask.squeeze() > 0.5).astype(bool)
        # compute IOU score by comparing the true mask to the predicted mask
        jaccard = jaccard_score(true_mask.flatten(), predicted_mask.flatten())
        train_scores.append(jaccard)

# compute IOU score using jaccard score for validation dataset
val_scores = []
for images, true_masks in val_dataset:
    predicted_masks = unet.predict(images)
    for true_mask, predicted_mask in zip(true_masks, predicted_masks):
        true_mask = true_mask.numpy().squeeze().astype(bool)  
        predicted_mask = (predicted_mask.squeeze() > 0.5).astype(bool)  
        jaccard = jaccard_score(true_mask.flatten(), predicted_mask.flatten())
        val_scores.append(jaccard)

# compute IOU score using jaccard score for test dataset
test_scores = []
best_worst = []
for images, true_masks in test_dataset:
    predicted_masks = unet.predict(images)
    for image, true_mask, predicted_mask in zip(images, true_masks, predicted_masks):
        true_mask = true_mask.numpy().squeeze().astype(bool)  
        predicted_mask = (predicted_mask.squeeze() > 0.5).astype(bool)  
        jaccard = jaccard_score(true_mask.flatten(), predicted_mask.flatten())
        test_scores.append(jaccard)
        best_worst.append((jaccard, image, true_mask, predicted_mask))

# compute mean Jaccard scores
mean_train_jaccard = np.mean(train_scores)
mean_val_jaccard = np.mean(val_scores)
mean_test_jaccard = np.mean(test_scores)

# print out train,validation,test performance
print("Train Accuracy:", train_accuracy)
print("Validation Accuracy:", val_accuracy)
print("Test Accuracy:", test_accuracy)
print("Train Loss:", train_loss)
print("Validation Loss:", val_loss)
print("Test Loss:", test_loss)

# print out mean IOU scores
print("Train Mean IOU:", mean_train_jaccard)
print("Validation Mean IOU:", mean_val_jaccard)
print("Test Mean IOU:", mean_test_jaccard)

# save figures of original image + mask + predicted mask
def plot_and_save_images(images, true_masks, predicted_masks, index, folder):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
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
    plt.savefig(os.path.join(folder, f"predicted_mask_{index}.png"))
    plt.close()

# get images and true masks from test dataset
for i, (images, true_masks) in enumerate(test_dataset):
    # predict mask
    predicted_masks = unet.predict(images)
    
    # plot and save images + true masks + predicted masks for some examples
    for j in range(min(len(images), 10)):
        plot_and_save_images(images, true_masks, predicted_masks, j, unet_results)

# create directory for best/worst images
result_folder = "3best_worst"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# sort list by IOU scores to get 3 best
top_3_highest = sorted(best_worst, key=lambda x: x[0], reverse=True)[:3]

# sort list by IOU scores to get 3 worst
bottom_3_lowest = sorted(best_worst, key=lambda x: x[0])[:3]

def save_best_worst(image, true_masks, predicted_masks, index, folder, name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(true_masks, cmap="gray")
    axes[1].set_title("True Mask")
    axes[1].axis("off")
    
    axes[2].imshow(predicted_masks, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{name}_{index}.png"))
    plt.close()

# save figures for 3 highest IOU scores
for i, (jaccard, image, true_mask, predicted_mask) in enumerate(top_3_highest):
    save_best_worst(image, true_mask, predicted_mask, i, "3best_worst", "best")

# save figures for 3 lowest IOU scores
for i, (jaccard, image, true_mask, predicted_mask) in enumerate(bottom_3_lowest):
    save_best_worst(image, true_mask, predicted_mask, i, "3best_worst", "worst")

# create directory for Grad-CAM images
heatmap_images_folder = "heatmap_images"
if not os.path.exists(heatmap_images_folder):
    os.makedirs(heatmap_images_folder)

# create explainer for GRAD-CAM
explainer = GradCAM()

# iterate through the 7 convs2d layers to get grad-cam activations
for i in range(7):
    # call to explain() method
    j = i + 1

    # get output of explainer using test data and specific conv2d layer
    output = explainer.explain((X_test, y_test), unet, None, f"conv2d_{j}")

    # save output in heatmap_images folder
    explainer.save(output, heatmap_images_folder, f'gradcam_conv2d_{j}.png')

end = time.time()

print(f"Total Runtime: {end-start} seconds")
