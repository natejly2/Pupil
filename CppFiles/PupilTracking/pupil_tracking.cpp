#include "pupil_tracking.h"

// Constructor
using namespace std;
using namespace cv;
using namespace Ort;

PupilTracker::PupilTracker(const std::string& model_path, int camera_id)
    : env(ORT_LOGGING_LEVEL_WARNING, "eye-mask"),
    session_options(),
    PXL(Size(1, 15), CV_64FC1),
    PYL(Size(1, 15), CV_64FC1),
    resized_eye(128, 128, CV_8UC1),
    eye_float(128, 128, CV_32FC1),
    mask(128, 128, CV_8U),
    inputDims{ 1, 128, 128, 1 },
    inputValues(1 * 128 * 128 * 1),
    memInfo(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
    frameIdx(0)
{
    initializeONNX(model_path);
    initializeCameraParameters(camera_id);
    initializeContainers();
}

// Public Methods
PupilTrackerOutput PupilTracker::getResults() {
	// Return the current results
    PupilTrackerOutput output;
    output.ellipse = transformed_ellipse;
    output.transformed_frame = transformed_frame.clone();
	return output;

}
void PupilTracker::processVideo(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Cannot open video: " << video_path << endl;
        return;
    }

    start_time = chrono::high_resolution_clock::now();
    cv::Mat frame;

    while (cap.read(frame)) {
        if (!processFrame(frame)) {
            break; // User pressed 'q'
        }
		PupilTrackerOutput output = getResults();
   //     printf("Frame: %d, Ellipse: Center(%.2f, %.2f), Size(%.2f, %.2f), Angle(%.2f)\n",
   //         frameIdx, output.ellipse.center.x, output.ellipse.center.y,
   //         output.ellipse.size.width, output.ellipse.size.height,
			//output.ellipse.angle);

    }

    cap.release();
    cv::destroyAllWindows();
    printPerformanceStats();
}

void PupilTracker::processCamera(int camera_id) {
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera: " << camera_id << endl;
        return;
    }

    start_time = chrono::high_resolution_clock::now();
    cv::Mat frame;

    while (cap.read(frame)) {
        if (!processFrame(frame)) {
            break; // User pressed 'q'
        }
    }

    cap.release();
    cv::destroyAllWindows();
    printPerformanceStats();
}

Mat PupilTracker::prepare_frame(const Mat& frame, bool top_half) {
    if (top_half) {
        return frame(Rect(0, 0, frame.cols, frame.rows / 2));
    }
    else {
        return frame(Rect(0, frame.rows / 2, frame.cols, frame.rows / 2));
    }
}

Mat PupilTracker::remove_bright_spots(Mat image, int threshold, int replace) {
    Mat mask = image < threshold;
    image.setTo(replace, ~mask);
    return image;
}

void PupilTracker::image_remap(Mat& Input, Mat& Output, Mat& PX, Mat& PY, uchar VAL) {
    Mat_<float> map_x(Input.size()), map_y(Input.size());

    float aaa[15];
    float bbb[15];

    for (int k = 0; k < PX.rows; k++) {
        aaa[k] = PX.at<double>(k, 0);
        bbb[k] = PY.at<double>(k, 0);
    }

    // Generate x and y 2D maps using polynomial coefficients
    for (int i = 0; i < map_x.rows; i++) {
        float y = i;
        for (int j = 0; j < map_x.cols; j++) {
            float x = j;
            float value_x;
            float value_y;
            value_x = aaa[0] + aaa[1] * x + aaa[2] * y;
            value_y = bbb[0] + bbb[1] * x + bbb[2] * y;
            if (PX.rows > 3) {
                value_x = value_x + aaa[3] * x * y + aaa[4] * x * x + aaa[5] * y * y;
                value_y = value_y + bbb[3] * x * y + bbb[4] * x * x + bbb[5] * y * y;
            }
            if (PX.rows > 6) {
                value_x = value_x + aaa[6] * x * x * y + aaa[7] * x * y * y + aaa[8] * x * x * x + aaa[9] * y * y * y;
                value_y = value_y + bbb[6] * x * x * y + bbb[7] * x * y * y + bbb[8] * x * x * x + bbb[9] * y * y * y;
            }
            if (PX.rows > 10) {
                value_x = value_x + (aaa[10] * x * x * x * y + aaa[11] * x * x * y * y + aaa[12] * x * y * y * y + aaa[13] * x * x * x * x + aaa[14] * y * y * y * y);
                value_y = value_y + (bbb[10] * x * x * x * y + bbb[11] * x * x * y * y + bbb[12] * x * y * y * y + bbb[13] * x * x * x * x + bbb[14] * y * y * y * y);
            }

            map_x.at<float>(i, j) = value_x;
            map_y.at<float>(i, j) = value_y;
        }
    }

    remap(Input, Output, map_x, map_y, INTER_CUBIC, BORDER_CONSTANT, Scalar(VAL, VAL, VAL));
}

tuple<Mat, int, int, int> PupilTracker::process_eye_crop(const Mat& frame, const vector<Rect>& eyes) {
    Rect eye_rect = eyes[0];
    int size = max(eye_rect.width, eye_rect.height);

    Rect crop_rect(eye_rect.x, eye_rect.y, size, size);

    // Ensure crop_rect is within frame bounds
    crop_rect &= Rect(0, 0, frame.cols, frame.rows);

    Mat eye_crop = frame(crop_rect).clone();
    eye_crop = remove_bright_spots(eye_crop, 220, 100);

    Mat eye_gray;
    cvtColor(eye_crop, eye_gray, COLOR_BGR2GRAY);

    return make_tuple(eye_gray, eye_rect.x, eye_rect.y, size);
}

bool PupilTracker::processFrame(cv::Mat& frame) {
    this->frameIdx++;

    frame = prepare_frame(frame, top_half);
    image_remap(frame, transformed_frame, PXL, PYL, 255);

    if (!checkFrameQuality(frame)) {
        cv::imshow("original", transformed_frame);
        return cv::waitKey(1) != 'q';
    }

    vector<cv::Rect> eyes = detectEyes(frame);
    if (eyes.empty()) {
        displayNoEyesFound();
        cv::imshow("original", transformed_frame);
        return cv::waitKey(1) != 'q';
    }

    auto [eye, x, y, sz] = process_eye_crop(frame, eyes);

    cv::Mat processed_mask = runInference(eye);

    processEllipseFitting(processed_mask, x, y, sz, transformed_frame);

    cv::imshow("original", transformed_frame);
    return cv::waitKey(1) != 'q';
}

// Getters and setters
void PupilTracker::setTopHalf(bool top_half_enabled) {
    top_half = top_half_enabled;
}

bool PupilTracker::getTopHalf() const {
    return top_half;
}

void PupilTracker::setSmoothingParameters(float x_a, float y_a, float w_a, float h_a, float r_a) {
    x_alpha = x_a;
    y_alpha = y_a;
    width_alpha = w_a;
    height_alpha = h_a;
    rotation_alpha = r_a;
}

int PupilTracker::getFrameCount() const {
    return frameIdx;
}

void PupilTracker::resetTracking() {
    prevEyes.clear();
    prevEllipse = cv::RotatedRect();
    frameIdx = 0;
}

// Private Methods
void PupilTracker::initializeONNX(const std::string& model_path) {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetExecutionMode(ORT_SEQUENTIAL);

    const ORTCHAR_T* model_path_ortchar =
#ifdef _WIN32
        std::wstring(model_path.begin(), model_path.end()).c_str();
#else
        model_path.c_str();
#endif

    session = std::make_unique<Ort::Session>(env, model_path_ortchar, session_options);

    // Get input/output names and store as strings to avoid AllocatedStringPtr issues
    auto inputNamePtr = session->GetInputNameAllocated(0, allocator);
    auto outputNamePtr = session->GetOutputNameAllocated(0, allocator);

    inputNameStr = std::string(inputNamePtr.get());
    outputNameStr = std::string(outputNamePtr.get());

    inputName = inputNameStr.c_str();
    outputName = outputNameStr.c_str();

    std::cout << "ONNX model loaded." << endl;
}

void PupilTracker::read_camera_polynomials(Mat& PX, Mat& PY, int vers) {
    float aaa[15];
    float bbb[15];
    if (vers == 0) {
        // OLD COEFFICIENTS FOR STEREO CAMERA
        aaa[0] = 49.3327484;
        aaa[1] = 0.313052624;
        aaa[2] = -0.268833011;
        aaa[3] = 0.000697896117;
        aaa[4] = 0.00263804453;
        aaa[5] = 0.000402396632;
        aaa[6] = 1.22087647e-06;
        aaa[7] = -1.78134894e-06;
        aaa[8] = -3.34541824e-06;
        aaa[9] = 6.22241657e-07;
        aaa[10] = -1.66110670e-09;
        aaa[11] = 7.09257042e-10;
        aaa[12] = -1.05600118e-09;
        aaa[13] = 8.04536882e-10;
        aaa[14] = -5.05777031e-10;

        bbb[0] = 39.9674644;
        bbb[1] = -0.286878079;
        bbb[2] = 0.617642760;
        bbb[3] = 0.00128830154;
        bbb[4] = 0.000474616652;
        bbb[5] = 0.00141545793;
        bbb[6] = -2.15545447e-06;
        bbb[7] = 8.81056678e-07;
        bbb[8] = -2.72812137e-08;
        bbb[9] = -2.85000988e-06;
        bbb[10] = 3.01577985e-10;
        bbb[11] = -1.67554082e-09;
        bbb[12] = 2.81702495e-10;
        bbb[13] = -3.02898331e-11;
        bbb[14] = 1.49710000e-09;
    }
    // NEW COEFFICIENTS FOR STEREO CAMERA CORRECTION
    else {
        aaa[0] = 51.8531113;
        aaa[1] = 0.431856513;
        aaa[2] = -0.234688565;
        aaa[3] = 0.000727739593;
        aaa[4] = 0.00177165726;
        aaa[5] = 0.000481563708;
        aaa[6] = 1.75251316e-07;
        aaa[7] = -2.32387106e-06;
        aaa[8] = -1.54169163e-06;
        aaa[9] = 1.10212386e-06;
        aaa[10] = -1.15611208e-10;
        aaa[11] = -2.96004554e-10;
        aaa[12] = 7.99325939e-10;
        aaa[13] = -2.61034749e-10;
        aaa[14] = -1.95401628e-09;

        bbb[0] = 12.6356163;
        bbb[1] = -0.223098874;
        bbb[2] = 0.825352252;
        bbb[3] = 0.00122782716;
        bbb[4] = 0.000314956123;
        bbb[5] = 0.000432871137;
        bbb[6] = -1.72658724e-06;
        bbb[7] = 3.80899365e-07;
        bbb[8] = 4.06708907e-08;
        bbb[9] = 6.59616603e-07;
        bbb[10] = -2.61479643e-10;
        bbb[11] = -4.32392039e-11;
        bbb[12] = -7.11899650e-10;
        bbb[13] = -3.07357980e-12;
        bbb[14] = -3.71893405e-09;
    }

    for (int k = 0; k < 15; k++) {
        PX.at<double>(k, 0) = aaa[k];
        PY.at<double>(k, 0) = bbb[k];
    }
}

void PupilTracker::initializeCameraParameters(int camera_id) {
    read_camera_polynomials(PXL, PYL, camera_id);
}

void PupilTracker::initializeContainers() {
    contours.reserve(10);
}

bool PupilTracker::checkFrameQuality(const cv::Mat& frame) {
    if (cv::mean(frame)[0] < 30.0f) {
        cv::putText(transformed_frame, "NO PERSON", { 50, 50 },
            cv::FONT_HERSHEY_SIMPLEX, 1, { 0, 0, 255 }, 2);
        printf("NO PERSON\n");
        return false;
    }
    return true;
}

vector<Rect> PupilTracker::coarse_find(const Mat& frame, Size min_size) {
    // this lambda is run once, the first time coarse_find is called
    static CascadeClassifier eye_cascade = []() {
        CascadeClassifier c;
        c.load("C:/Desktop/Pupil/PythonFiles/Media/haarcascade_eye.xml");
        if (c.empty()) {
            throw std::runtime_error("Failed to load eye cascade");
        }
        return c;
        }();

    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    vector<Rect> eyes;
    eye_cascade.detectMultiScale(
        gray, eyes,
        1.05, 3, 0, min_size
    );
    return eyes;
}

vector<cv::Rect> PupilTracker::detectEyes(const cv::Mat& frame) {
    vector<cv::Rect> eyes;

    if (frameIdx % 1 == 0) {
        eyes = coarse_find(frame);
        if (!eyes.empty()) {
            prevEyes = eyes;
        }
        else if (!prevEyes.empty()) {
            eyes = prevEyes;
        }
    }
    else {
        eyes = prevEyes;
    }

    return eyes;
}

void PupilTracker::displayNoEyesFound() {
    cv::putText(transformed_frame, "No Eyes", { 50, 50 },
        cv::FONT_HERSHEY_SIMPLEX, 1, { 0, 0, 255 }, 2);
    printf("NO EYES\n");
}

cv::Mat PupilTracker::runInference(const cv::Mat& eye) {
    cv::resize(eye, resized_eye, { 128, 128 }, 0, 0, cv::INTER_LINEAR);
    resized_eye.convertTo(eye_float, CV_32F, 1.0 / 255.0f);

    memcpy(inputValues.data(), eye_float.data, inputValues.size() * sizeof(float));

    Value inputTensor = Value::CreateTensor<float>(
        memInfo, eye_float.ptr<float>(), inputValues.size(),
        inputDims.data(), inputDims.size()
    );

    auto outputTensors = session->Run(
        RunOptions{ nullptr },
        &inputName, &inputTensor, 1,
        &outputName, 1
    );

    float* outData = outputTensors.front().GetTensorMutableData<float>();
    cv::threshold(cv::Mat(128, 128, CV_32F, outData), mask, 0.5, 255, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);

    return mask;
}

RotatedRect PupilTracker::fit_ellipse_custom(const vector<vector<Point>>& contour, int bias_factor) {
    if (contour.empty()) return RotatedRect();

    vector<Point> all_pts;
    for (const auto& cnt : contour) {
        all_pts.insert(all_pts.end(), cnt.begin(), cnt.end());
    }

    if (all_pts.size() < 5) return RotatedRect();

    // Calculate mean y for biasing
    float mean_y = 0;
    for (const Point& pt : all_pts) {
        mean_y += pt.y;
    }
    mean_y /= all_pts.size();

    vector<Point> weighted_pts = all_pts;

    if (bias_factor > 0) {
        vector<Point> bottom_pts;
        for (const Point& pt : all_pts) {
            if (pt.y > mean_y) {
                bottom_pts.push_back(pt);
            }
        }

        for (int i = 0; i < bias_factor; i++) {
            weighted_pts.insert(weighted_pts.end(), bottom_pts.begin(), bottom_pts.end());
        }
    }

    return fitEllipse(weighted_pts);
}

RotatedRect PupilTracker::check_flip(RotatedRect ellipse) {
    float w = ellipse.size.width;
    float h = ellipse.size.height;
    float ang = ellipse.angle;

    if (w < h) {
        swap(w, h);
        ang += 90;
    }

    if (ang >= 90) {
        ang -= 180;
    }
    else if (ang < -90) {
        ang += 180;
    }

    ellipse.size.width = w;
    ellipse.size.height = h;
    ellipse.angle = ang;

    return ellipse;
}

cv::RotatedRect PupilTracker::apply_smoothing(
    const cv::RotatedRect& best_ellipse,
    float x, float y,  
    float x_alpha,
    float y_alpha,
    float width_alpha,
    float height_alpha,
    float rotation_alpha
) {
    cv::RotatedRect flipped = check_flip(best_ellipse);

    if (this->prevEllipse.size.width == 0 || this->prevEllipse.size.height == 0) {
        printf("No previous EMA, using current ellipse\n");
        this->prevEllipse = flipped;
        return flipped;
    }

    cv::Mat current = (cv::Mat_<float>(5, 1) <<
        flipped.center.x,
        flipped.center.y,
        flipped.size.width,
        flipped.size.height,
        flipped.angle);

    cv::Mat previous = (cv::Mat_<float>(5, 1) <<
        this->prevEllipse.center.x,
        this->prevEllipse.center.y,
        this->prevEllipse.size.width,
        this->prevEllipse.size.height,
        this->prevEllipse.angle);

    cv::Mat alphas = (cv::Mat_<float>(5, 1) <<
        x_alpha, y_alpha, width_alpha, height_alpha, rotation_alpha);

    cv::Mat one_minus_alpha = cv::Mat::ones(5, 1, CV_32F) - alphas;
    cv::Mat smoothed = alphas.mul(current) + one_minus_alpha.mul(previous);

	cv::RotatedRect result = 
        cv::RotatedRect(
        cv::Point2f(smoothed.at<float>(0), smoothed.at<float>(1)),
        cv::Size2f(smoothed.at<float>(2), smoothed.at<float>(3)),
        smoothed.at<float>(4));

    this->prevEllipse = result; 
    return result;
}

cv::RotatedRect PupilTracker::recalculate_ellipse(
    const cv::RotatedRect& ellipse,
    const cv::Mat& PX,
    const cv::Mat& PY
) {
    // take initial ellipse parameters xc,yc,rx,ry,eangle
    //  apply zoom transform given by PX, PY and return new ellipse parameters
    // rxc,ryc,rex,rey,reangle

    // unpack the incoming RotatedRect
    float xc = ellipse.center.x;
    float yc = ellipse.center.y;
    float rx = ellipse.size.width * 0.5f;
    float ry = ellipse.size.height * 0.5f;
    float eangle = ellipse.angle;

    int NE = 120;

    float  theta = eangle * CV_PI / 180.f;          // radians
    float  cosT = std::cos(theta);
    float  sinT = std::sin(theta);

    std::vector<cv::Point2f> srcPts;
    srcPts.reserve(NE);

    for (int k = 0; k < NE; ++k) {
        float t = 2.0f * CV_PI * k / NE;          // parametric angle
        float ct = std::cos(t);
        float st = std::sin(t);

        // rotated point
        float x = xc + rx * ct * cosT - ry * st * sinT;
        float y = yc + rx * ct * sinT + ry * st * cosT;

        srcPts.emplace_back(x, y);
    }

    constexpr int   MAX_ITERS = 6;      // Newton iterations
    constexpr float EPS = 1e-3f;       // step for finite-difference Jacobian
    constexpr float TOL = 1e-4f;       // convergence threshold

    const int order = PX.rows;         // 3 → up to x,y | 6 → up to x²,y² | …

    /* --------------------------------------------------------------------- */
    /* 1. copy polynomial coefficients into flat float[15] arrays            */
    /* --------------------------------------------------------------------- */
    float a[15] = { 0 }, b[15] = { 0 };
    for (int k = 0; k < order; ++k) {
        a[k] = static_cast<float>(PX.at<double>(k, 0));
        b[k] = static_cast<float>(PY.at<double>(k, 0));
    }

    /* --------------------------------------------------------------------- */
    /* 2. helper lambdas: evaluate poly & (numerical) partial derivatives    */
    /* --------------------------------------------------------------------- */
    auto evalPoly = [order](float x, float y, const float* c) -> float {
        float v = c[0] + c[1] * x + c[2] * y;
        if (order > 3)
            v += c[3] * x * y + c[4] * x * x + c[5] * y * y;
        if (order > 6)
            v += c[6] * x * x * y + c[7] * x * y * y
            + c[8] * x * x * x + c[9] * y * y * y;
        if (order > 10)
            v += c[10] * x * x * x * y + c[11] * x * x * y * y
            + c[12] * x * y * y * y + c[13] * x * x * x * x
            + c[14] * y * y * y * y;
        return v;
        };

    auto deriv = [&](float x, float y,
        const float* c,
        float& dfdx, float& dfdy) {
            // simple central differences: df/dx, df/dy
            dfdx = (evalPoly(x + EPS, y, c) - evalPoly(x - EPS, y, c)) / (2 * EPS);
            dfdy = (evalPoly(x, y + EPS, c) - evalPoly(x, y - EPS, c)) / (2 * EPS);
        };

    /* --------------------------------------------------------------------- */
    /* 3. Newton inversion loop for each contour point                       */
    /* --------------------------------------------------------------------- */
    std::vector<cv::Point2f> dstPts(srcPts.size());

    for (size_t i = 0; i < srcPts.size(); ++i) {
        // (xd,yd) = distorted – the point we *have*
        const float xd = srcPts[i].x;
        const float yd = srcPts[i].y;

        // start guess: assume distortion is small → xd,yd already close
        float xu = xd;
        float yu = yd;

        for (int it = 0; it < MAX_ITERS; ++it) {
            // forward map at current guess
            const float Fx = evalPoly(xu, yu, a) - xd;
            const float Fy = evalPoly(xu, yu, b) - yd;

            if (std::fabs(Fx) + std::fabs(Fy) < TOL) break;   // converged

            float dFxdx, dFxdy, dFydx, dFydy;
            deriv(xu, yu, a, dFxdx, dFxdy);
            deriv(xu, yu, b, dFydx, dFydy);

            const float det = dFxdx * dFydy - dFxdy * dFydx;
            if (std::fabs(det) < 1e-12f) break;               // singular

            // Newton step: [xu,yu] ← [xu,yu] − J⁻¹·F
            xu -= (dFydy * Fx - dFxdy * Fy) / det;
            yu -= (-dFydx * Fx + dFxdx * Fy) / det;
        }

        dstPts[i].x = xu;            // now UNDISTORTED
        dstPts[i].y = yu;
    }

    // pack up the new ellipse
    cv::RotatedRect e2 = cv::fitEllipse(dstPts);   // needs ≥5 pts

    float rxc = e2.center.x;
    float ryc = e2.center.y;
    float rex = e2.size.width * 0.5f;
    float rey = e2.size.height * 0.5f;
    float reangle = e2.angle;

    return cv::RotatedRect(
        cv::Point2f(rxc, ryc),
        cv::Size2f(rex * 2.0f, rey * 2.0f),
        reangle
    );
}

cv::RotatedRect PupilTracker::processEllipseFitting(const cv::Mat& processed_mask, int x, int y, int sz, cv::Mat& transformed_frame) {
    contours.clear();
    cv::findContours(processed_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        // Find largest contour
        size_t maxIdx = 0;
        double maxArea = 0;
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                maxIdx = i;
            }
        }

        if (contours[maxIdx].size() >= 5) {
            vector<vector<cv::Point>> best_contour;
            best_contour.push_back(contours[maxIdx]);

            cv::RotatedRect ellipse = fit_ellipse_custom(best_contour);

            // Rescale back to original frame coords
            float scale = float(sz) / 128.f;
            ellipse.center.x = ellipse.center.x * scale + x;
            ellipse.center.y = ellipse.center.y * scale + y;
            ellipse.size.width *= scale;
            ellipse.size.height *= scale;

			// print ellipse for debugging



            RotatedRect smooth_ellipse = apply_smoothing(
                ellipse, static_cast<float>(x), static_cast<float>(y),
                 x_alpha, y_alpha, width_alpha, height_alpha, rotation_alpha
            );


			// print prevEllipse for debugging

            // Draw results
            transformed_ellipse = recalculate_ellipse(smooth_ellipse, PXL, PYL);

            cv::ellipse(transformed_frame, transformed_ellipse, { 0, 255, 0 }, 2);
            cv::circle(transformed_frame,
                { int(transformed_ellipse.center.x), int(transformed_ellipse.center.y) },
                3, { 0, 0, 255 }, -1);
            return transformed_ellipse;

        }

    }
    else {
        cv::putText(transformed_frame, "No Eyes", { 100, 300 },
            cv::FONT_HERSHEY_SIMPLEX, 1, { 0, 0, 255 }, 2);
        printf("NO EYES\n");
    }
}

void PupilTracker::printPerformanceStats() {
    auto end = chrono::high_resolution_clock::now();
    double secs = chrono::duration<double>(end - start_time).count();

    std::cout << "Processed " << frameIdx
        << " frames in " << secs << "s ("
        << (frameIdx / secs) << " FPS)\n";
}
