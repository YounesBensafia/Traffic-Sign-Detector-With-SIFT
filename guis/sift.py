import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def find_best_match(query_image_path, images_folder_path):
    img1 = cv2.imread(query_image_path, 0)
    sift = cv2.SIFT_create(nfeatures=5000)
    bf = cv2.BFMatcher()

    best_match = None
    max_good_matches = 0

    kp1, des1 = sift.detectAndCompute(img1, None)
    if des1 is None:
        print("Pas de descripteurs pour l'image de requête. Veuillez vérifier l'image.")
        return None

    for filename in os.listdir(images_folder_path):
        if filename.endswith('.png'):
            img2 = cv2.imread(os.path.join(images_folder_path, filename), 0)

            kp2, des2 = sift.detectAndCompute(img2, None)
            if des2 is None:
                print(f"Pas de descripteurs pour l'image {filename}. Ignorée.")
                continue

            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:  
                    good.append([m])

            if len(good) > max_good_matches:
                max_good_matches = len(good)
                best_match = filename

    # Afficher le meilleur match
    if best_match:
        print(f"Meilleur match : {best_match} avec {max_good_matches} correspondances.")
        img2 = cv2.imread(os.path.join(images_folder_path, best_match), 0)
        kp2, des2 = sift.detectAndCompute(img2, None)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append([m])
        # Dessiner les correspondances
        img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_matches)
        plt.title('Correspondances')
        plt.show()
    else:
        print("Aucun match significatif trouvé.")

# Example usage
find_best_match('inputs/image3.png', 'images')
