import cv2

# Load the image
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
	print("Error: Could not load image.")
	exit()

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(image, None)

# Print the descriptors
print("Descriptors:\n", descriptors)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with keypoints
cv2.imshow('SIFT Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()