import os
import shutil

# Paths for source folder, destination folder, and txt file
gt_folder = '/home/hung/Downloads/CASIA 1.0 groundtruth/CM'
image_folder = '/home/hung/Downloads/archive/CASIA1/Sp'
destination_folder = '/home/hung/Downloads/CASIA 1.0 groundtruth/ImageClass'
txt_file_path = './train.txt'

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Open the txt file to save filenames
with open(txt_file_path, 'w') as txt_file:
    for filename in os.listdir(gt_folder):
        if filename.endswith('_gt.png'):
            # Get the corresponding image filename
            base_filename = filename.replace('_gt.png', '.jpg')
            gt_filepath = os.path.join(gt_folder, filename)
            image_filepath = os.path.join(image_folder, base_filename)
            print(image_filepath)
            # Move both ground truth and corresponding image to destination
            if os.path.exists(image_filepath):
                # shutil.move(gt_filepath, destination_folder)
                shutil.copy(image_filepath, destination_folder)
                
                # Write the corresponding image filename to txt file
                txt_file.write(f"{base_filename}\n")

print(f"All files have been moved, and filenames saved in {txt_file_path}")
