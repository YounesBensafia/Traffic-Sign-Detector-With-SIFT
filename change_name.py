import os

def rename_images(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out non-image files (optional)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # Sort the files (optional)
    image_files.sort()
    
    # Rename each file
    for i, filename in enumerate(image_files):
        # Create the new filename with .png extension
        new_filename = f"{i}.png"
        
        # Get the full path to the old and new filenames
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        
        # Rename the file if the new filename does not already exist
        if not os.path.exists(new_file):
            os.rename(old_file, new_file)
            print(f"Renamed '{filename}' to '{new_filename}'")
        else:
            print(f"Skipped renaming '{filename}' to '{new_filename}' because it already exists")

# Specify the folder containing the images
folder_path = 'images'

# Call the function to rename the images
rename_images(folder_path)
