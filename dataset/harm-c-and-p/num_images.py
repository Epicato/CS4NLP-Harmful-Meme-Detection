import os

# Path to the directory containing the images
images_dir = "./harm-p/images"

# List all files in the directory
all_files = os.listdir(images_dir)

# Filter out PNG files
png_files = [file for file in all_files if file.endswith('.png')]

# Count the number of PNG files
num_png_files = len(png_files)

print(f"Number of PNG files in '{images_dir}': {num_png_files}")
