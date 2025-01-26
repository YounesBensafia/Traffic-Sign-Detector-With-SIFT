import os

def rename_files(folder_path):
    files = os.listdir(folder_path)
    for i, filename in enumerate(files):
        if filename.endswith(".jpg"):
            new_name = f"{i}.jpg"
            new_path = os.path.join(folder_path, new_name)
            if not os.path.exists(new_path):
                os.rename(os.path.join(folder_path, filename), new_path)

if __name__ == "__main__":
    folder_path = "images"
    rename_files(folder_path)