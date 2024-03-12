import os
import matplotlib.pyplot as plt
from PIL import Image

# paths to data
images_folder = 'models/data/images'
masks_folder = 'models/data/masks'
output_folder = 'eda_visuals'

# list of images/masks
image_files = sorted(os.listdir(images_folder))
mask_files = sorted(os.listdir(masks_folder))

# print number of images/masks
num_images = len(image_files)
num_masks = len(mask_files)

print(f"Number of images: {num_images}")
print(f"Number of masks: {num_masks}\n")

# load images/masks
def load_images_and_masks(image_files, mask_files):
    images = []
    masks = []
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(images_folder, img_file)
        mask_path = os.path.join(masks_folder, mask_file)
        image = Image.open(img_path)
        mask = Image.open(mask_path)  
        images.append(image)
        masks.append(mask)
    return images, masks

# load images and masks
images, masks = load_images_and_masks(image_files, mask_files)

# plot each pair of image and mask in one figure
num_examples = 10
for i in range(num_examples):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(images[i])
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    axes[1].imshow(masks[i])
    axes[1].set_title('Mask')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'image_mask_{i+1}.png'))
    plt.close()

# get unique dimensions of images to see difference in image sizes
def get_unique_dimensions(image_files, mask_files):
    dimensions = set()
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(images_folder, img_file)
        mask_path = os.path.join(masks_folder, mask_file)
        image_size = Image.open(img_path).size
        mask_size = Image.open(mask_path).size
        dimensions.add(image_size)
        dimensions.add(mask_size)
    return dimensions

# get unique dimensions
unique_dimensions = get_unique_dimensions(image_files, mask_files)

# print unique dimensions
print("Unique dimensions of images and masks:")
for dim in unique_dimensions:
    print(dim)