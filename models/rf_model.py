# import libraries
import os
import time
import numpy as np
from PIL import Image
from preprocess import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, jaccard_score

# get the current directory of file
script_dir = os.path.dirname(os.path.abspath(__file__))

# get the parent directory of the data folder
images_folder = "data/images"
masks_folder = "data/masks"

# load and preprocess images and masks
images, masks = load_and_preprocess_data(images_folder, masks_folder, 30)

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
