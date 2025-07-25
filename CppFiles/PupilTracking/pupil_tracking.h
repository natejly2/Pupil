#pragma once

#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <memory>
#include <tuple>
#include <array>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

/**
 * @brief Structure to hold eye crop results
 */
struct EyeCropResult {
    cv::Mat gray;
    int x, y, size;
};

struct PupilTrackerOutput {
    cv::RotatedRect ellipse;
    cv::Mat transformed_frame;
};
/**
 * @brief Real-time pupil tracking class using ONNX neural network and OpenCV
 *
 * This class provides functionality for detecting and tracking pupils in video streams
 * or static images using a combination of Haar cascades for eye detection and a
 * neural network for precise pupil segmentation.
 */
class PupilTracker {
public:

	//returns PupilTrackerOutput containing the transformed ellipse and frame
    PupilTrackerOutput getResults();

    /**
     * @brief Constructor
     * @param model_path Path to the ONNX model file for pupil segmentation
     * @param camera_id Camera ID for polynomial correction coefficients (0 or 1)
     */
    PupilTracker(const std::string& model_path, int camera_id = 1);

    /**
     * @brief Destructor
     */
    ~PupilTracker() = default;

    // Delete copy constructor and assignment operator
    PupilTracker(const PupilTracker&) = delete;
    PupilTracker& operator=(const PupilTracker&) = delete;

    /**
     * @brief Process a video file for pupil tracking
     * @param video_path Path to the video file
     */
    void processVideo(const std::string& video_path);

    /**
     * @brief Process live camera feed for pupil tracking
     * @param camera_id Camera device ID (default: 0)
     */
    void processCamera(int camera_id = 0);

    /**
     * @brief Process a single frame
     * @param frame Input frame to process
     * @return true if processing should continue, false if user requested exit
     */
    bool processFrame(cv::Mat& frame);

    // Configuration methods
    /**
     * @brief Enable/disable processing only the top half of frames
     * @param top_half_enabled If true, only process top half of frame
     */
    void setTopHalf(bool top_half_enabled);

    /**
     * @brief Get current top half processing setting
     * @return true if top half processing is enabled
     */
    bool getTopHalf() const;

    /**
     * @brief Set smoothing parameters for ellipse tracking
     * @param x_a X-coordinate smoothing factor (0.0 - 1.0)
     * @param y_a Y-coordinate smoothing factor (0.0 - 1.0)
     * @param w_a Width smoothing factor (0.0 - 1.0)
     * @param h_a Height smoothing factor (0.0 - 1.0)
     * @param r_a Rotation smoothing factor (0.0 - 1.0)
     */
    void setSmoothingParameters(float x_a, float y_a, float w_a, float h_a, float r_a);

    /**
     * @brief Get current frame count
     * @return Number of frames processed
     */
    int getFrameCount() const;

    /**
     * @brief Reset tracking state
     */
    void resetTracking();

private:
    // ONNX Runtime components
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string inputNameStr;
    std::string outputNameStr;
    const char* inputName;
    const char* outputName;
    Ort::MemoryInfo memInfo;


    // Transformed frame for display
    cv::Mat transformed_frame;
    //Transformed ellipse for display
    cv::RotatedRect transformed_ellipse;



    // Camera correction parameters
    cv::Mat PXL;  ///< X-direction polynomial coefficients
    cv::Mat PYL;  ///< Y-direction polynomial coefficients

    // Smoothing parameters
    float x_alpha;
    float y_alpha;
    float width_alpha;
    float height_alpha;
    float rotation_alpha;
    bool top_half;

    // Tracking state
    std::vector<cv::Rect> prevEyes;
    cv::RotatedRect prevEllipse;

    // Working matrices (preallocated for performance)
    cv::Mat resized_eye;
    cv::Mat eye_float;
    cv::Mat mask;

    // Preallocated containers
    std::vector<std::vector<cv::Point>> contours;
    std::array<int64_t, 4> inputDims;
    std::vector<float> inputValues;

    // Performance tracking
    std::chrono::high_resolution_clock::time_point start_time;
    int frameIdx;

    // Private methods

    /**
     * @brief Initialize ONNX Runtime session
     * @param model_path Path to ONNX model file
     */
    void initializeONNX(const std::string& model_path);

    /**
     * @brief Initialize camera polynomial correction parameters
     * @param camera_id Camera version ID for coefficient selection
     */
    void initializeCameraParameters(int camera_id);

    /**
     * @brief Initialize preallocated containers
     */
    void initializeContainers();

    /**
     * @brief Read camera polynomial coefficients
     * @param PX Output matrix for X coefficients
     * @param PY Output matrix for Y coefficients
     * @param vers Camera version (0 for old, 1 for new coefficients)
     */
    void read_camera_polynomials(cv::Mat& PX, cv::Mat& PY, int vers);

    /**
     * @brief Prepare frame for processing (crop top/bottom half)
     * @param frame Input frame
     * @param top_half If true, return top half, otherwise bottom half
     * @return Cropped frame
     */
    cv::Mat prepare_frame(const cv::Mat& frame, bool top_half);

    /**
     * @brief Remove bright spots from image
     * @param image Input image
     * @param threshold Brightness threshold (default: 200)
     * @param replace Replacement value (default: 0)
     * @return Processed image
     */
    cv::Mat remove_bright_spots(cv::Mat image, int threshold = 200, int replace = 0);

    /**
     * @brief Apply polynomial transformation to image
     * @param Input Source image
     * @param Output Destination image
     * @param PX X-direction polynomial coefficients
     * @param PY Y-direction polynomial coefficients
     * @param VAL Fill value for out-of-bounds pixels
     */
    void image_remap(cv::Mat& Input, cv::Mat& Output, cv::Mat& PX, cv::Mat& PY, uchar VAL);

    /**
     * @brief Process eye crop from detected eye region
     * @param frame Input frame
     * @param eyes Detected eye rectangles
     * @return Tuple containing (processed_eye_image, x, y, size)
     */
    std::tuple<cv::Mat, int, int, int> process_eye_crop(const cv::Mat& frame, const std::vector<cv::Rect>& eyes);

    /**
     * @brief Check frame quality (brightness, presence detection)
     * @param frame Input frame
     * @return true if frame quality is acceptable
     */
    bool checkFrameQuality(const cv::Mat& frame);

    /**
     * @brief Coarse eye detection using Haar cascades
     * @param frame Input frame
     * @param min_size Minimum detection size
     * @return Vector of detected eye rectangles
     */
    std::vector<cv::Rect> coarse_find(const cv::Mat& frame, cv::Size min_size = cv::Size(200, 200));

    /**
     * @brief Detect eyes in frame using cached results when appropriate
     * @param frame Input frame
     * @return Vector of eye rectangles
     */
    std::vector<cv::Rect> detectEyes(const cv::Mat& frame);

    /**
     * @brief Display "No Eyes" message on frame
     */
    void displayNoEyesFound();

    /**
     * @brief Run neural network inference for pupil segmentation
     * @param eye Cropped eye image
     * @return Binary mask of pupil region
     */
    cv::Mat runInference(const cv::Mat& eye);

    /**
     * @brief Fit ellipse to contour with optional bottom bias
     * @param contour Input contours
     * @param bias_factor Bias factor for bottom weighting (-1 for no bias)
     * @return Fitted rotated rectangle (ellipse)
     */
    cv::RotatedRect fit_ellipse_custom(const std::vector<std::vector<cv::Point>>& contour, int bias_factor = -1);

    /**
     * @brief Normalize ellipse orientation
     * @param ellipse Input ellipse
     * @return Normalized ellipse with consistent orientation
     */
    cv::RotatedRect check_flip(cv::RotatedRect ellipse);

    /**
     * @brief Apply exponential moving average smoothing to ellipse parameters
     * @param best_ellipse Current frame ellipse
     * @param x X offset for coordinate adjustment
     * @param y Y offset for coordinate adjustment
     * @param ema Previous smoothed ellipse (modified in-place)
     * @param x_alpha X smoothing factor
     * @param y_alpha Y smoothing factor
     * @param width_alpha Width smoothing factor
     * @param height_alpha Height smoothing factor
     * @param rotation_alpha Rotation smoothing factor
     * @return Smoothed ellipse
     */
    cv::RotatedRect apply_smoothing(
        const cv::RotatedRect& best_ellipse,
        float x, float y,
        float x_alpha, float y_alpha, float width_alpha, float height_alpha, float rotation_alpha
    );

    /**
     * @brief Transform ellipse coordinates using inverse polynomial mapping
     * @param ellipse Input ellipse in distorted coordinates
     * @param PX X-direction polynomial coefficients
     * @param PY Y-direction polynomial coefficients
     * @return Ellipse in corrected coordinate system
     */
    cv::RotatedRect recalculate_ellipse(
        const cv::RotatedRect& ellipse,
        const cv::Mat& PX,
        const cv::Mat& PY
    );

    /**
     * @brief Process ellipse fitting from segmentation mask
     * @param processed_mask Binary pupil mask
     * @param x X offset of eye crop
     * @param y Y offset of eye crop
     * @param sz Size of eye crop
     */
    cv::RotatedRect processEllipseFitting(const cv::Mat& processed_mask, int x, int y, int sz, cv::Mat& transformed_frame);
 
    /**
     * @brief Print performance statistics
     */
    void printPerformanceStats();
};