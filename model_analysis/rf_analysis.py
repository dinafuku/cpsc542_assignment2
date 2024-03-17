import os
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, jaccard_score
import joblib

# get data directory respectively
parent_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(parent_dir, '..', 'models', 'rf_model.pkl')

rf_model = joblib.load(model_dir)

# Load the training and testing data
data_dir = os.path.join(parent_dir, '..', 'models', 'rf_data.npz')
data = np.load(data_dir)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

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

# plotting and saving original image, true mask, predicted mask for image segmentation mask
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

# create directory for rf images
rf_results = "rf_results"
if not os.path.exists(rf_results):
    os.makedirs(rf_results)

# grab test set example
sample_index = 0
sample_image = X_test[sample_index]
sample_true_mask = y_test[sample_index]

# predict mask
sample_pred_mask = rf_model.predict(sample_image.reshape(1, -1))

# plot and save
plot_rf_result(sample_image, sample_true_mask, sample_pred_mask, rf_results, sample_index)