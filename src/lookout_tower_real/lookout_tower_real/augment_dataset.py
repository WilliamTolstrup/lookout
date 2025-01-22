import cv2
import os
import random
import albumentations as A

# Define paths
input_folder = '/home/william/Datasets/floorspace'
output_folder = '/home/william/Datasets/floorspace_augmented'
os.makedirs(output_folder, exist_ok=True)

# Define a list of potential augmentations
augmentation_choices = [
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Transpose(),
    A.RandomBrightnessContrast(),
    A.RandomGamma(),
    A.HueSaturationValue(),
    A.Resize(256, 256)  # Resize all images to a consistent size
]

# Helper function to apply random augmentations and save images
def apply_random_augmentations(image, master_image, index, floor_id):
    # Randomly select 3-5 augmentations from the list
    num_augmentations = random.randint(3, 5)
    random_augmentations = random.sample(augmentation_choices, num_augmentations)
    
    # Apply the selected augmentations to both the image and the master
    augmentation_pipeline = A.Compose(random_augmentations)
    augmented = augmentation_pipeline(image=image, mask=master_image)
    
    augmented_image = augmented['image']
    augmented_master = augmented['mask']

    # Save the augmented image and master pair
    augmented_image_path = os.path.join(output_folder, f'{floor_id}_{index}_augmented.png')
    augmented_master_path = os.path.join(output_folder, f'{floor_id}_{index}_master_augmented.png')

    # Ensure the augmented image and master are numpy arrays for OpenCV
    augmented_image = augmented_image.astype('uint8')
    augmented_master = augmented_master.astype('uint8')

    # Save the augmented image and master
    cv2.imwrite(augmented_image_path, augmented_image)
    cv2.imwrite(augmented_master_path, augmented_master)
    
    print(f"Augmented {floor_id}_{index} and saved as {augmented_image_path}, {augmented_master_path}")

# Loop through the dataset
for floor_id in range(1, 3):  # Change 1 to 3 if there are more floors
    for i in range(1, 17):  # Loop through floor1_1 to floor1_16
        if floor_id == 1 and i <= 16:
            image_path = os.path.join(input_folder, f'floor1_{i}.png')
            master_image_path = os.path.join(input_folder, f'floor1_master.png')
        elif floor_id == 2 and i <= 15:
            image_path = os.path.join(input_folder, f'floor2_{i}.png')
            master_image_path = os.path.join(input_folder, f'floor2_master.png')

        if os.path.exists(image_path):
            # Read images
            image = cv2.imread(image_path)
            master_image = cv2.imread(master_image_path)

            # Apply augmentations multiple times
            for j in range(1, 11):  # Apply 5 random augmentations per image
                apply_random_augmentations(image, master_image, f'{i}_{j}', f'floor{floor_id}')
