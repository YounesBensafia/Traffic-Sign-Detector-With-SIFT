import cv2
import os
import matplotlib.pyplot as plt

# Load the reference image
img1 = cv2.imread('image.png')
if img1 is None:
    raise FileNotFoundError("Reference image not found. Check the path 'images/6.jpg'.")

folder_path = 'images'

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)

if descriptors1 is None:
    raise ValueError("No keypoints detected in the reference image.")

best_match_count = 0
best_match_image = None
best_matches = None
best_keypoints2 = None

for filename in os.listdir(folder_path):
    if filename.endswith('4.jpg') or filename.endswith('4.png'):
        img_path = os.path.join(folder_path, filename)

        img2 = cv2.imread(img_path)
        if img2 is None:
            print(f"Skipping {filename}: Unable to read image.")
            continue
        
        keypoints2, descriptors2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
        if descriptors2 is None:
            print(f"Skipping {filename}: No keypoints detected.")
            continue
        
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) > best_match_count:
            best_match_count = len(matches)
            best_match_image = img2
            best_matches = matches
            best_keypoints2 = keypoints2

if best_match_image is not None and best_keypoints2 is not None:
    img_matches = cv2.drawMatches(
        img1, keypoints1, best_match_image, best_keypoints2, best_matches, None
    )
    
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'Best Match ({best_match_count} Matches Found)')
    plt.axis('off')
    plt.show()
    
else:
    print("No matches found.")
