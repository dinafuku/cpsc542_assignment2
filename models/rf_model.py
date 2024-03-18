# import libraries
import os
import time
import numpy as np
from PIL import Image
from preprocess import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, jaccard_score
import matplotlib.pyplot as plt
import joblib

# get data directory respectively
parent_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(parent_dir, '..', 'data', 'images')
mask_dir = os.path.join(parent_dir, '..', 'data', 'masks')

# load and preprocess images and masks
images, masks = load_and_preprocess_data(image_dir, mask_dir, 250)

# flatten images and masks for RF
X_train_flat = images.reshape(len(images), -1)  
y_train_flat = masks.reshape(len(masks), -1)    

# TTS (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_train_flat, y_train_flat, test_size=0.2, random_state=542)

np.savez("rf_data.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# create/fit RF model
rf_model = RandomForestClassifier(n_estimators=1, random_state=542)
fit_start = time.time()
rf_model.fit(X_train, y_train)
fit_end = time.time()
print(f"Fitting took {fit_end-fit_start} seconds...")

# save the trained RF model
joblib.dump(rf_model, "rf_model.pkl")