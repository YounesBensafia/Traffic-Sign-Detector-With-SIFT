import cv2 
import time

sift = cv2.SIFT_create()

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    suc, img1 = cap.read()
    if not suc:
        break

    img2 = cv2.imread('inputs/image1.png', 0)

    if img2 is None:
        print("Image not found!")
        break

    start = time.time()

    # Convert to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors using SIFT
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_gray, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # Match descriptors using the brute force matcher
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    # Draw the first 100 matches
    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:100], img2, flags=2)

    # Display FPS on the image
    cv2.putText(img3, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Show the result
    cv2.imshow('SIFT Matches', img3)

    # Break on 'ESC' key press
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
