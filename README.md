
---

# Feature Detection and Matching in Stereo Images

## Overview

This project involves feature detection and matching in stereo images using the Middlebury dataset. It applies Lowe's ratio test to filter matches and estimates the camera pose. Additionally, it evaluates the accuracy of pose estimation and explores methods to detect and remove moving objects.

## Steps and Algorithms

### 1. Loading and Preprocessing the Dataset
- **Dataset**: Stereo images from the Middlebury dataset.
- **Preprocessing**: Images are converted to grayscale to simplify feature detection.

### 2. Feature Detection and Description
- **Algorithm**: Scale-Invariant Feature Transform (SIFT).
- **Purpose**: Detects and describes features robustly to scale and rotation variations.

### 3. Feature Matching
- **Matcher**: Fast Library for Approximate Nearest Neighbors (FLANN).
- **Purpose**: Efficiently finds nearest neighbors for feature points.

### 4. Lowe’s Ratio Test
- **Purpose**: Filters out poor matches by comparing the distance of the closest match to the second-closest match. Retains matches where the closest match is significantly closer.

### 5. Camera Pose Estimation
- **Method**: Compute the essential matrix using filtered matches.
- **Decomposition**: Estimates the rotation and translation between the two camera views.

### 6. Error Metrics
- **Evaluation**: Accuracy of the camera pose estimation is assessed by comparing estimated rotation and translation with ground truth values. Errors in rotation and translation are computed.

### 7. Detecting and Removing Moving Objects (Bonus)
- **Techniques**: Background subtraction to identify and mask moving objects, improving the robustness of the feature matching process.

## Results and Visualizations

### 1. Feature Detection and Matching
- **Visualization**: SIFT keypoints detected and matched between stereo images. Significant number of good matches were retained after applying Lowe’s ratio test.

### 2. Camera Pose Estimation
- **Outcome**: Essential matrix computed successfully; camera pose estimated and compared with ground truth values.

### 3. Error Metrics
- **Results**: Rotation and translation errors were calculated, showing that the estimated pose was close to the ground truth.

### 4. Removing Moving Objects
- **Preprocessing**: Background subtraction improved feature matching by reducing false matches caused by moving objects.

## Discussion and Improvements

### 1. Improving Feature Detection and Matching
- **Advanced Methods**: Exploring feature detectors like ORB or deep learning-based methods.
- **Constraints**: Incorporating epipolar geometry to refine matching and reduce outliers.

### 2. Refining Camera Pose Estimation
- **Motion Models**: Using sophisticated models that account for scene structure and motion dynamics.
- **Camera Calibration**: Accurate calibration minimizes intrinsic parameter errors, enhancing pose estimation.

### 3. Outlier Rejection Techniques
- **Methods**: Employing RANSAC and other outlier rejection methods to improve pose estimation accuracy.

### 4. Handling Moving Objects
- **Advanced Detection**: Utilizing deep learning models like YOLO or Mask R-CNN for better moving object detection and removal.

## Conclusion

The implemented feature detection, matching, and camera pose estimation pipeline demonstrated robust performance on the Middlebury dataset. Evaluating error metrics and exploring methods to handle moving objects revealed several avenues for improvement. Future work should focus on integrating advanced feature detectors, refining matching techniques, and enhancing outlier rejection methods for better accuracy.

## Visualizations and Graphs

- **Feature Matches**: Visualizations of feature matches between stereo images before and after applying Lowe’s ratio test.
- **Error Metrics**: Graphs showing the rotation and translation errors between the estimated and ground truth camera poses.
- **Moving Object Detection**: Images showing results of background subtraction and moving object removal.



---

