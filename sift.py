import cv2
import os
import matplotlib.pyplot as plt

import cv2
import os
import numpy as np

def find_best_match(input_image_path, repo_path):
    img_input = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img_input is None:
        raise FileNotFoundError(f"Input image {input_image_path} not found.")
    
    sift = cv2.SIFT_create()
    _, descriptors_input = sift.detectAndCompute(img_input, None)
    
    bf = cv2.BFMatcher()
    best_match = None
    max_matches = 0
    
    for file_name in os.listdir(repo_path):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_repo = cv2.imread(os.path.join(repo_path, file_name), cv2.IMREAD_GRAYSCALE)
        if img_repo is None:
            continue
            
        _, descriptors_repo = sift.detectAndCompute(img_repo, None)
        if descriptors_repo is None:
            continue

        matches = bf.knnMatch(descriptors_input, descriptors_repo, k=2)
        good_matches = []
        
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good_matches.append(m)
        
        num_matches = len(good_matches)
        # IL AFFICHE HENAYA ALL THE MATCHES
        print(f"{file_name}: {num_matches} matches")
        
        if num_matches > max_matches:
            max_matches = num_matches
            best_match = file_name

    return best_match, max_matches


input_image = "input.png"  
# repo_folder = "data_to_use/" 
repo_folder = "images/" 

best_match, max_matches = find_best_match(input_image, repo_folder)
print(best_match, max_matches)


# POUR AFFICHER BRQ L'IMAGE
if best_match:
    img_input = cv2.imread(input_image)
    img_best_match = cv2.imread(os.path.join(repo_folder, best_match))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img_best_match, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Best Match Image")
    axes[1].axis('off')

    plt.show()
else:
    print("No match found.")
    print("No match found.")