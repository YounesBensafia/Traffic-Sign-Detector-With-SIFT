import cv2
import numpy as np
import os

image_folder = './data_to_use'
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
i = len(image_files)+2


image = cv2.imread("./images/image.png")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0) 

low_threshold = 10
high_threshold = 50 
edges = cv2.Canny(blurred_image, low_threshold, high_threshold)



contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

filtered_contours = []
for contour in contours:
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    is_valid_contour = False
    
    if len(approx) == 3:
        area = cv2.contourArea(contour)
        if area >= 300:  
            filtered_contours.append(contour)
            is_valid_contour = True
            x, y, w, h = cv2.boundingRect(contour)
            y1, y2 = max(0, y-10), min(image.shape[0], y+h+10)
            x1, x2 = max(0, x-10), min(image.shape[1], x+w+10)
            roi = image[y1:y2, x1:x2]
            cv2.imwrite(f'./data_to_use/contour_{i}.png', roi)
            i=i+1
    elif len(approx) == 8: 
        area = cv2.contourArea(contour)
        if area >= 300:
            filtered_contours.append(contour)
            x, y, w, h = cv2.boundingRect(contour)
            y1, y2 = max(0, y-10), min(image.shape[0], y+h+10)
            x1, x2 = max(0, x-10), min(image.shape[1], x+w+10)
            roi = image[y1:y2, x1:x2]

            cv2.imwrite(f'./data_to_use/contour_{i}.png', roi)
            i=i+1
            is_valid_contour = True
    
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if area < 300:
        continue
    

    if perimeter != 0: 
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        if 0.8 < circularity < 1.2: 
            if not is_valid_contour:
                filtered_contours.append(contour)
                x, y, w, h = cv2.boundingRect(contour)
                y1, y2 = max(0, y-10), min(image.shape[0], y+h+10)
                x1, x2 = max(0, x-10), min(image.shape[1], x+w+10)
                roi = image[y1:y2, x1:x2]
                cv2.imwrite(f'./data_to_use/contour_{i}.png', roi)
                i=i+1
                is_valid_contour = True







# cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)

# cv2.imshow('Contours', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



