#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Function to perform feature detection and matching
void detectAndMatch(const Mat& img1, const Mat& img2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<DMatch>& good_matches) {
    // Feature detection using ORB
    Ptr<FeatureDetector> detector = ORB::create();
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

    // Feature description
    Ptr<DescriptorExtractor> extractor = ORB::create();
    Mat descriptors1, descriptors2;
    extractor->compute(img1, keypoints1, descriptors1);
    extractor->compute(img2, keypoints2, descriptors2);

    // Feature matching
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<vector<DMatch>> matches;
    matcher->knnMatch(descriptors1, descriptors2, matches, 2);

    // Apply Lowe's ratio test
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < 0.7 * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }
}

// Function to estimate camera pose
Mat estimatePose(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, const vector<DMatch>& good_matches, const Mat& K) {
    vector<Point2f> points1, points2;
    for (size_t i = 0; i < good_matches.size(); i++) {
        points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }

    Mat E = findEssentialMat(points1, points2, K, RANSAC);
    Mat R, t;
    recoverPose(E, points1, points2, K, R, t);

    Mat pose = Mat::eye(4, 4, CV_64F);
    R.copyTo(pose(Rect(0, 0, 3, 3)));
    t.copyTo(pose(Rect(3, 0, 1, 3)));

    return pose;
}

// Function to calculate error metrics
void calculateError(const Mat& estimated_pose, const Mat& ground_truth_pose) {
    Mat R_est = estimated_pose(Rect(0, 0, 3, 3));
    Mat t_est = estimated_pose(Rect(3, 0, 1, 3));

    Mat R_gt = ground_truth_pose(Rect(0, 0, 3, 3));
    Mat t_gt = ground_truth_pose(Rect(3, 0, 1, 3));

    // Calculate rotation error
    Mat R_diff = R_est.t() * R_gt;
    double trace = R_diff.at<double>(0,0) + R_diff.at<double>(1,1) + R_diff.at<double>(2,2);
    double rot_error = acos((trace - 1) / 2) * 180 / CV_PI;

    // Calculate translation error
    double trans_error = norm(t_est - t_gt);

    cout << "Rotation error: " << rot_error << " degrees" << endl;
    cout << "Translation error: " << trans_error << " units" << endl;
}

int main() {
    try {
        cout << "Starting program..." << endl;

        // Load stereo images
        Mat img1 = imread("left_image.png", IMREAD_COLOR);
        Mat img2 = imread("right_image.png", IMREAD_COLOR);

        if (img1.empty() || img2.empty()) {
            cout << "Error loading images" << endl;
            return -1;
        }

        cout << "Images loaded successfully" << endl;

        // Convert to grayscale for feature detection
        Mat gray1, gray2;
        cvtColor(img1, gray1, COLOR_BGR2GRAY);
        cvtColor(img2, gray2, COLOR_BGR2GRAY);

        // Detect and match features
        vector<KeyPoint> keypoints1, keypoints2;
        vector<DMatch> good_matches;
        detectAndMatch(gray1, gray2, keypoints1, keypoints2, good_matches);

        cout << "Feature detection and matching completed" << endl;
        cout << "Number of good matches: " << good_matches.size() << endl;

        // Draw matches
        Mat img_matches;
        drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                    Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Show matches
        namedWindow("Matches", WINDOW_NORMAL);
        imshow("Matches", img_matches);

        // Camera intrinsic matrix (this should be calibrated for your specific camera)
        Mat K = (Mat_<double>(3,3) << 
            718.856, 0, 607.1928,
            0, 718.856, 185.2157,
            0, 0, 1);

        // Estimate camera pose
        Mat estimated_pose = estimatePose(keypoints1, keypoints2, good_matches, K);

        cout << "Camera pose estimated" << endl;

        // Load ground truth pose (this should be provided with your dataset)
        Mat ground_truth_pose = Mat::eye(4, 4, CV_64F);
        // TODO: Load actual ground truth pose data here
        cout << "Warning: Using identity matrix as ground truth pose. Replace with actual data." << endl;

        // Calculate error
        calculateError(estimated_pose, ground_truth_pose);

        cout << "Program completed successfully" << endl;

        waitKey(0); // Wait for a key press
    }
    catch (const exception& e) {
        cerr << "An exception occurred: " << e.what() << endl;
    }
    catch (...) {
        cerr << "An unknown exception occurred" << endl;
    }

    system("pause"); // Keep console window open
    return 0;
}
