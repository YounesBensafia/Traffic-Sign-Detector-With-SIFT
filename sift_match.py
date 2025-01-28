import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Load the query image in grayscale
img1 = cv2.imread('inputs/image.png', cv2.IMREAD_GRAYSCALE)

# Create a mask for the region of interest in the query image
mask = np.zeros(img1.shape, dtype=np.uint8)
roi = cv2.selectROI("Select ROI", img1)  # Interactively select the ROI
cv2.destroyWindow("Select ROI")
mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 255

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for the query image within the mask
kp1, des1 = sift.detectAndCompute(img1, mask)

# Folder containing the images to compare
folder_path = 'images'

# Initialize variables to track the best match
best_match = None
max_good_matches = 0
best_img = None
best_kp = None
best_matches = None

# Iterate over all images in the folder
for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    
    # Skip non-image files
    if not (filepath.endswith('.png') or filepath.endswith('.jpg') or filepath.endswith('.jpeg')):
        continue
    
    # Load the current image in grayscale
    img2 = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img2 is None:
        continue

    # Detect keypoints and compute descriptors for the current image
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Initialize the BFMatcher
    bf = cv2.BFMatcher()

    # Match descriptors using knn
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Update the best match if the current image has more good matches
    if len(good_matches) > max_good_matches:
        max_good_matches = len(good_matches)
        best_match = filename
        best_img = img2
        best_kp = kp2
        best_matches = good_matches

# Draw the matches with the best image
if best_img is not None:
    img_matches = cv2.drawMatches(img1, kp1, best_img, best_kp, best_matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the best match
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'Best Match: {best_match} ({max_good_matches} good matches)')
    plt.axis('off')
    plt.show()
else:
    print("No matching images found in the folder.")
