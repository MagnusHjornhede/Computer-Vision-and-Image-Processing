import cv2
import os
import time
import matplotlib.pyplot as plt

# get image pairs from folders
def get_image_pairs(left_folder, right_folder):
    left_images = sorted(os.listdir(left_folder))
    right_images = sorted(os.listdir(right_folder))
    return [(os.path.join(left_folder, l), os.path.join(right_folder, r)) for l, r in zip(left_images, right_images)]

# matching between image pairs
def feature_matching(image_pair, feature_detector, matcher):
    img1 = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)

    # Detect and compute features
    kp1, des1 = feature_detector.detectAndCompute(img1, None)
    kp2, des2 = feature_detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None: # If nothing found
        return 0, None  # None detected

    # Match features
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    return len(good_matches), good_matches

# feature matching on dataset and calculate performance metrics
def process_dataset(left_folder, right_folder):
    image_pairs = get_image_pairs(left_folder, right_folder)

    # Initialize detector and matcher
    feature_detector = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    total_matches = 0
    failed_matches = 0
    times = []

    for idx, image_pair in enumerate(image_pairs):
        start_time = time.time()
        num_matches, good_matches = feature_matching(image_pair, feature_detector, bf)
        end_time = time.time()

        # calculate time per frame
        times.append(end_time - start_time)

        if num_matches == 0:
            failed_matches += 1
        total_matches += num_matches

        print(f"Processed Image Pair {idx + 1}: {num_matches} matches")

    avg_fps = len(image_pairs) / sum(times)
    avg_matches = total_matches / len(image_pairs)

    return avg_fps, avg_matches, failed_matches

#  visualize and save the N-th image pair matching
def visualize_nth_pair(left_folder, right_folder, N, feature_detector, matcher, output_path):
    image_pairs = get_image_pairs(left_folder, right_folder)

    if N < len(image_pairs):
        # Load the N-th image pair
        img1_path, img2_path = image_pairs[N]
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        # Detect and compute features
        kp1, des1 = feature_detector.detectAndCompute(img1, None)
        kp2, des2 = feature_detector.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            print("No features detected for this pair.")
            return

        # Match features
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # Visualize the matching
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Save the matched image
        cv2.imwrite(output_path, img_matches)

        # Show the matches
        plt.figure(figsize=(12, 6))
        plt.imshow(img_matches)
        plt.title(f'Feature Matching for {N + 1}-th Image Pair')
        plt.show()
    else:
        print(f"N is out of range. The dataset contains {len(image_pairs)} image pairs.")

# Process custom dataset
custom_left_folder = "custom_dataset/left"
custom_right_folder = "custom_dataset/right"

print("Processing Custom Dataset...")
fps_custom, avg_matches_custom, failed_matches_custom = process_dataset(custom_left_folder, custom_right_folder)
print(f"Custom Dataset - FPS: {fps_custom}, Avg Matches: {avg_matches_custom}, Failed Matches: {failed_matches_custom}")

# Process PennCOSYVIO dataset
penn_left_folder = "PennCOSYVIO_dataset/left_cam_frames"
penn_right_folder = "PennCOSYVIO_dataset/right_cam_frames"

print("\nProcessing PennCOSYVIO Dataset...")
fps_penn, avg_matches_penn, failed_matches_penn = process_dataset(penn_left_folder, penn_right_folder)
print(f"PennCOSYVIO Dataset - FPS: {fps_penn}, Avg Matches: {avg_matches_penn}, Failed Matches: {failed_matches_penn}")

# Visualize and save multiple pairs
for N in [4, 9, 14, 19]:  # 5th, 10th, 15th, and 20th pairs
    print(f"Visualizing {N + 1}-th Image Pair in Custom Dataset...")
    visualize_nth_pair(custom_left_folder, custom_right_folder, N, cv2.ORB_create(),
                       cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
                       f'output_images/pair_{N + 1}_matches_custom.png')

    print(f"Visualizing {N + 1}-th Image Pair in PennCOSYVIO Dataset...")
    visualize_nth_pair(penn_left_folder, penn_right_folder, N, cv2.ORB_create(),
                       cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
                       f'output_images/pair_{N + 1}_matches_penn.png')
