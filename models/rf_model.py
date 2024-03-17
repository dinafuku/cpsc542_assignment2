# import libraries
import os
import time
import numpy as np
from PIL import Image
from preprocess import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, jaccard_score

# get data directory respectively
parent_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(parent_dir, '..', 'data', 'images')
mask_dir = os.path.join(parent_dir, '..', 'data', 'masks')

# load and preprocess images and masks
images, masks = load_and_preprocess_data(image_dir, mask_dir, 30)

# flatten images and masks for RF
X_train_flat = images.reshape(len(images), -1)  
y_train_flat = masks.reshape(len(masks), -1)    

# TTS (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_train_flat, y_train_flat, test_size=0.2, random_state=542)

# create/fit RF model
rf_model = RandomForestClassifier(n_estimators=1, random_state=542)
fit_start = time.time()
rf_model.fit(X_train, y_train)
fit_end = time.time()
print(f"Fitting took {fit_end-fit_start} seconds...")

# predict masks for train and test data
train_pred_masks = rf_model.predict(X_train)
test_pred_masks = rf_model.predict(X_test)

# calculate train acc
train_accuracy = accuracy_score(y_train.flatten(), train_pred_masks.flatten())

# calculate IOU score for training data using jaccard_score
train_iou = jaccard_score(y_train.flatten(), train_pred_masks.flatten(), average='macro')

# calculate test acc
test_accuracy = accuracy_score(y_test.flatten(), test_pred_masks.flatten())

# calculate IOU score for testing data using jaccard_score
test_iou = jaccard_score(y_test.flatten(), test_pred_masks.flatten(), average='macro')

# print out results
print("\nTraining Accuracy: ", train_accuracy)
print("Testing Accuracy: ", test_accuracy)
print("Training IoU: ", train_iou)
print("Testing IoU: ", test_iou)

# plotting and saving original image, true make, predicted mask for image segmentation mask
def plot_rf_result(image, true_mask, predicted_mask, folder, index):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(image.reshape(224, 224, 3))
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(true_mask.reshape(224, 224), cmap="gray")
    axes[1].set_title("True Mask")
    axes[1].axis("off")
    
    axes[2].imshow(predicted_mask.reshape(224, 224), cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"rf_result_{index}.png"))
    plt.close()

# grabbing an example from test set
sample_index = 0
sample_image = X_test[sample_index]
sample_true_mask = y_test[sample_index]

# predict mask
sample_pred_mask = rf_model.predict(sample_image.reshape(1, -1))

# plot and save
plot_rf_result(sample_image, sample_true_mask, sample_pred_mask, "rf_results", sample_index)
