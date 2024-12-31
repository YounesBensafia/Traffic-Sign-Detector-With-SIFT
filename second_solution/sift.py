import numpy as np
import cv2
from matplotlib import pyplot as plt

def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    equalized = cv2.equalizeHist(blurred)
    
    filtered = cv2.bilateralFilter(equalized, 9, 75, 75)
    
    return filtered

# Read images
img1 = cv2.imread('image.png', 0)     
img2 = cv2.imread('images/image.png', 0)        


img1 = preprocess_image(img1)
img2 = preprocess_image(img2)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
print(des1[0], kp1[0].pt)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test with slightly stricter threshold
good = []
for m, n in matches:
    # Reduced threshold from 0.75 to 0.7 for more strict matching
    if m.distance < 0.7 * n.distance:
        print(m.queryIdx, m.trainIdx, m.distance, n.queryIdx, n.trainIdx, n.distance, n.imgIdx)
        good.append([m])

# Draw matches
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

plt.imshow(img3)
plt.show()