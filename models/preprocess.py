import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# preprocess and load data for RF and CNN model
def load_and_preprocess_data(images_folder, masks_folder, num_images=100, img_size=(224, 224)):
    # get sorted list of images/masks
    image_files = sorted(os.listdir(images_folder))[:num_images]
    mask_files = sorted(os.listdir(masks_folder))[:num_images]
    
    images = []
    masks = []

    # iterate through pairs of images, masks
    for img_file, mask_file in zip(image_files, mask_files):
        # create full paths
        img_path = os.path.join(images_folder, img_file)
        mask_path = os.path.join(masks_folder, mask_file)
        # load and preprocess images
        image = Image.open(img_path).resize(img_size)
        mask = Image.open(mask_path).resize(img_size).convert('L')
        # append images and masks to lists as numpy arrays
        images.append(np.array(image))
        masks.append(np.array(mask))
    return np.array(images), np.array(masks)