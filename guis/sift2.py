import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def find_best_match(input_image_path, repo_path, min_matches=10, ratio_thresh=0.7):
    # Read image in color for visualization
    img_input_color = cv2.imread(input_image_path)
    if img_input_color is None:
        raise FileNotFoundError(f"Input image {input_image_path} not found.")

    # Convert to grayscale for SIFT
    img_input = cv2.cvtColor(img_input_color, cv2.COLOR_BGR2GRAY)

    # Get the base filename of the input image
    input_filename = os.path.basename(input_image_path)

    sift = cv2.SIFT_create()
    kp_input, descriptors_input = sift.detectAndCompute(img_input, None)

    bf = cv2.BFMatcher()
    best_match = None
    max_matches = 0
    best_matches = None
    best_kp = None
    best_img = None

    for file_name in os.listdir(repo_path):
        # Skip if it's not an image file
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Skip if it's the same image as input
        if file_name == input_filename:
            continue

        # Read repo image in color
        img_repo_color = cv2.imread(os.path.join(repo_path, file_name))
        if img_repo_color is None:
            continue

        # Convert to grayscale for SIFT
        img_repo = cv2.cvtColor(img_repo_color, cv2.COLOR_BGR2GRAY)

        kp_repo, descriptors_repo = sift.detectAndCompute(img_repo, None)
        if descriptors_repo is None:
            continue

        matches = bf.knnMatch(descriptors_input, descriptors_repo, k=2)
        good_matches = []

        # Filter matches using ratio test
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        if len(good_matches) < min_matches:
            continue

        good_matches_flat = [match[0] for match in good_matches]

        # Get matched points coordinates
        src_pts = np.float32(
            [kp_input[m.queryIdx].pt for m in good_matches_flat]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp_repo[m.trainIdx].pt for m in good_matches_flat]
        ).reshape(-1, 1, 2)

        # Calculate the center of matched points
        center_dst = np.mean(dst_pts.reshape(-1, 2), axis=0)

        # Calculate distances from each point to the center
        distances = np.linalg.norm(dst_pts.reshape(-1, 2) - center_dst, axis=1)

        # Define a radius threshold (adjust this value based on your images)
        radius_threshold = min(img_repo.shape) * 0.2  # 20% of smaller image dimension

        # Count points within the radius
        points_within_radius = np.sum(distances < radius_threshold)

        # Calculate clustering ratio
        clustering_ratio = points_within_radius / len(distances)

        # Calculate final score combining number of matches and clustering
        score = len(good_matches) * clustering_ratio

        print(
            f"{file_name}: {len(good_matches)} matches, clustering ratio: {clustering_ratio:.2f}, score: {score:.2f}"
        )

        if (
            score > max_matches and clustering_ratio > 0.6
        ):  # Add minimum clustering threshold
            max_matches = score
            best_match = file_name
            best_matches = good_matches
            best_kp = kp_repo
            best_img = img_repo_color

    return (
        best_match,
        max_matches,
        kp_input,
        best_kp,
        best_matches,
        img_input_color,
        best_img,
    )


if __name__ == "__main__":

    input_image = "input.png"
    repo_folder = "images/"

    (
        best_match,
        max_matches,
        kp_input,
        kp_best,
        good_matches,
        img_input_color,
        img_best,
    ) = find_best_match(input_image, repo_folder)
    print(f"Best match: {best_match} with {max_matches} matches")

    if best_match:
        # Create figure with larger size
        plt.figure(figsize=(15, 10))

        # Draw matches
        img_matches = cv2.drawMatchesKnn(
            img_input_color,
            kp_input,
            img_best,
            kp_best,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Show matches (images will now be in color)
        plt.subplot(211)
        plt.imshow(
            cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
        )  # Convert BGR to RGB for matplotlib
        plt.title("Keypoint Matches")
        plt.axis("off")

        # Show original images with keypoints (in color)
        plt.subplot(223)
        img_kp1 = cv2.drawKeypoints(
            img_input_color,
            kp_input,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        plt.imshow(cv2.cvtColor(img_kp1, cv2.COLOR_BGR2RGB))
        plt.title("Input Image Keypoints")
        plt.axis("off")

        plt.subplot(224)
        img_kp2 = cv2.drawKeypoints(
            img_best, kp_best, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        plt.imshow(cv2.cvtColor(img_kp2, cv2.COLOR_BGR2RGB))
        plt.title("Best Match Keypoints")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    else:
        print("No match found.")
