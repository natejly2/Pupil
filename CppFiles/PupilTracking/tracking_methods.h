#ifndef EYE_TRACKER_H
#define EYE_TRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace cv;
using namespace std;

// STRUCTS AND GLOBAL VARIABLES
struct EllipseData {
    RotatedRect ellipse;
    int x, y;
};

extern CascadeClassifier eye_cascade;
extern Mat mask_global;
extern vector<Rect> prev_eyes;
extern EllipseData prev_ellipse;
extern vector<int> thresholds;


// FUNCTIONS
/**
 * Initialize the eye tracker - loads cascade classifier and sets up global variables
 * @return nothing
 */
void initialize_eye_tracker();

/**
 * Detect eye regions in frame using Haar cascade
 * @param frame Input frame
 * @return Vector of detected eye rectangles
 */
vector<Rect> coarse_find(const Mat& frame, Size min_size = Size(200, 200));

/**
 * Remove bright spots from image by setting pixels above threshold to replace value
 * @param image Input image
 * @param threshold Brightness threshold (default: 200)
 * @param replace Replacement value (default: 0)
 * @return Processed image
 */
Mat remove_bright_spots(Mat image, int threshold = 200, int replace = 0);

/**
 * Find the darkest area in image using grid-based search
 * @param image Input grayscale image
 * @return Pair of darkest rectangle and its mean value
 */
pair<Rect, double> find_dark_area(const Mat& image);

/**
 * Generate multiple threshold images using different threshold values
 * @param image Input grayscale image
 * @param dark_point Base threshold value
 * @return Vector of thresholded binary images
 */
vector<Mat> threshold_images(const Mat& image, double dark_point);

/**
 * Extract and filter contours from binary images
 * @param images Vector of binary images
 * @param min_area Minimum contour area (default: 1500)
 * @param margin Border margin to exclude contours (default: 3)
 * @return Pair of filtered contours and contour visualization images
 */
pair<vector<vector<vector<Point>>>, vector<Mat>> get_contours(const vector<Mat>& images,
    int min_area = 1500, int margin = 3);

/**
 * Fit ellipse to contour points with optional bottom bias
 * @param contour Vector of contour point sets
 * @param bias_factor Factor for biasing towards bottom points (default: -1, no bias)
 * @return Fitted rotated rectangle (ellipse)
 */
RotatedRect fit_ellipse_custom(const vector<vector<Point>>& contour, int bias_factor = -1);

/**
 * Normalize ellipse orientation and angle
 * @param ellipse Input ellipse
 * @return Normalized ellipse with consistent orientation
 */
RotatedRect check_flip(RotatedRect ellipse);

/**
 * Prepare frame by cropping to top or bottom half based on TOP flag
 * @param frame Input frame
 * @return Cropped frame
 */
Mat prepare_frame(const Mat& frame, bool top_half);

/**
 * Extract and process eye crop from frame
 * @param frame Input frame
 * @param eyes Vector of detected eye rectangles
 * @return Tuple of (processed eye image, x offset, y offset, size)
 */
tuple<Mat, int, int, int> process_eye_crop(const Mat& frame, const vector<Rect>& eyes);

/**
 * Generate ellipse candidates from eye image
 * @param eye_gray Grayscale eye image
 * @param dark_val Dark threshold value
 * @return Tuple of (threshold images, contour images, ellipse images, ellipses)
 */
tuple<vector<Mat>, vector<Mat>, vector<Mat>, vector<RotatedRect>>
generate_ellipse_candidates(const Mat& eye_gray, double dark_val);

/**
 * Calculate quality scores for ellipse candidates
 * @param thresholded_images Vector of thresholded images
 * @param ellipses Vector of ellipse candidates
 * @return Vector of quality scores for each ellipse
 */
vector<double> calculate_ellipse_scores(const vector<Mat>& thresholded_images,
    const vector<RotatedRect>& ellipses);

/**
 * Select best ellipse from candidates with temporal consistency checks
 * @param ellipses Vector of ellipse candidates
 * @param percents Vector of quality scores
 * @param x X offset
 * @param y Y offset
 * @param frame_idx Current frame index
 * @return Tuple of (best ellipse, updated x, updated y)
 */
tuple<RotatedRect, int, int> select_best_ellipse(const vector<RotatedRect>& ellipses,
    const vector<double>& percents, int x, int y, int frame_idx);

/**
 * Apply exponential moving average smoothing to ellipse parameters
 * @param best_ellipse Input ellipse
 * @param x X offset
 * @param y Y offset
 * @return Smoothed ellipse in full frame coordinates
 */
RotatedRect apply_smoothing(const RotatedRect& best_ellipse, float x, float y, RotatedRect ema, float x_alpha, float y_alpha, float width_alpha, float height_alpha, float rotation_alpha);
/**
 * Display tracking results with visualization grids
 * @param frame Output frame for drawing
 * @param thresholded_images Vector of threshold images for grid display
 * @param contour_images Vector of contour images for grid display
 * @param ellipse_images Vector of ellipse images for grid display
 * @param full_ellipse Final smoothed ellipse
 * @param center Center point of detected ellipse
 * @param x X offset
 * @param y Y offset
 * @param frame_idx Current frame index
 */
void display_results(Mat& frame, const vector<Mat>& thresholded_images,
    const vector<Mat>& contour_images, const vector<Mat>& ellipse_images,
    const RotatedRect& full_ellipse, const Point2f& center,
    int x, int y, int frame_idx);

/**
 * Main eye tracking function - processes video file
 * @param video_path Path to input video file
 */
void run_eye_tracking(const string& video_path);

#endif // EYE_TRACKER_H