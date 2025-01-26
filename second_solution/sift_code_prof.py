import numpy as np
import cv2
from matplotlib import pyplot as plt

# Code from:
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

# Prepared for M1IV students, by Prof. Slimane Larabi
#===================================================

img1 = cv2.imread('image.png',0)          # queryImage
img2 = cv2.imread('images/image.png',0) # trainImage

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
print(des1[0],kp1[0].pt)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# Sort them in the order of their distance.
#matches = sorted(matches, key = lambda x:x.distance)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        print(m.queryIdx, m.trainIdx, m.distance, n.queryIdx, n.trainIdx, n.distance, n.imgIdx)
        good.append([m])
        
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None, flags=2)

plt.imshow(img3),plt.show()