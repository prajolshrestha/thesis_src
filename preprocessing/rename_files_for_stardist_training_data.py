import os

# Define the paths to the training and testing directories
training_masks_path = 'training_data/train/masks'
testing_masks_path = 'training_data/test/masks'

# Function to rename files
def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('_masks.tif'):
            new_filename = filename.replace('_masks.tif', '.tif')
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f'Renamed: {filename} to {new_filename}')

# Rename files in the training and testing masks directories
rename_files(training_masks_path)
rename_files(testing_masks_path)

print("Renaming complete.")
