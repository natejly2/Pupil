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

struct EllipseData {
    RotatedRect ellipse;
    int x, y;
};

// Global variables for state tracking
CascadeClassifier eye_cascade;
vector<Rect> prev_eyes;
EllipseData prev_ellipse;
bool has_prev_ellipse = false;
vector<int> thresholds = { 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48 };

void initialize_eye_tracker() {
    if (!eye_cascade.load("C:/Desktop/Pupil/haarcascade_eye.xml")) {
        cerr << "Error loading cascade\n";
    }
}

vector<Rect> coarse_find(const Mat& frame, Size min_size = Size(200, 200)) {
    // this lambda is run once, the first time coarse_find is called
    static CascadeClassifier eye_cascade = []() {
        CascadeClassifier c;
        c.load("C:/Desktop/Pupil/haarcascade_eye.xml");
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


Mat remove_bright_spots(Mat image, int threshold = 200, int replace = 0) {
    Mat mask = image < threshold;
    image.setTo(replace, ~mask);
    return image;
}

pair<Rect, double> find_dark_area(const Mat& image) {
    int num_grids = 9;
    int h = image.rows;
    int w = image.cols;
    int grid_h = h / num_grids;
    int grid_w = w / num_grids;

    double darkest_val = 255.0;
    Rect darkest_square;

    for (int i = 0; i < num_grids; i++) {
        for (int j = 0; j < num_grids; j++) {
            Rect roi(j * grid_w, i * grid_h, grid_w, grid_h);
            Mat grid = image(roi);
            Scalar mean_val = mean(grid);

            if (mean_val[0] < darkest_val) {
                darkest_val = mean_val[0];
                darkest_square = roi;
            }
        }
    }

    return make_pair(darkest_square, darkest_val);
}

vector<Mat> threshold_images(const Mat& image, double dark_point) {
    vector<Mat> images;
    Mat denoised;
    GaussianBlur(image, denoised, Size(5, 5), 0);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    for (int t : thresholds) {
        Mat binary;
        threshold(denoised, binary, dark_point + t, 255, THRESH_BINARY_INV);

        Mat opened;
        morphologyEx(binary, opened, MORPH_OPEN, kernel, Point(-1, -1), 1);

        Mat mask = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
        Mat flood = opened.clone();
        floodFill(flood, mask, Point(0, 0), Scalar(255));

        Mat flood_inv;
        bitwise_not(flood, flood_inv);

        Mat filled;
        bitwise_or(opened, flood_inv, filled);

        images.push_back(filled);
    }

    return images;
}

pair<vector<vector<vector<Point>>>, vector<Mat>> get_contours(const vector<Mat>& images,
    int min_area = 1500, int margin = 3) {
    vector<vector<vector<Point>>> filtered_contours;
    vector<Mat> contour_images;

    for (const Mat& img : images) {
        vector<vector<Point>> contours;
        findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<vector<Point>> kept;
        for (const auto& cnt : contours) {
            if (contourArea(cnt) < min_area) continue;

            bool touches_border = false;
            for (const Point& pt : cnt) {
                if (pt.x < margin || pt.x > img.cols - margin ||
                    pt.y < margin || pt.y > img.rows - margin) {
                    touches_border = true;
                    break;
                }
            }

            if (!touches_border) {
                kept.push_back(cnt);
            }
        }

        // Sort by area and keep only the largest
        sort(kept.begin(), kept.end(), [](const vector<Point>& a, const vector<Point>& b) {
            return contourArea(a) > contourArea(b);
            });

        if (kept.size() > 1) {
            kept.resize(1);
        }

        // Create convex hull
        vector<Point> hull_points;
        if (!kept.empty()) {
            for (const auto& cnt : kept) {
                hull_points.insert(hull_points.end(), cnt.begin(), cnt.end());
            }

            vector<Point> hull;
            convexHull(hull_points, hull);
            kept = { hull };
        }

        Mat ci = Mat::zeros(img.size(), CV_8UC1);
        if (!kept.empty()) {
            drawContours(ci, kept, -1, Scalar(255), 2);
        }

        filtered_contours.push_back(kept);
        contour_images.push_back(ci);
    }

    return make_pair(filtered_contours, contour_images);
}

RotatedRect fit_ellipse_custom(const vector<vector<Point>>& contour, int bias_factor = -1) {
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

RotatedRect check_flip(RotatedRect ellipse) {
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

Mat prepare_frame(const Mat& frame, bool top_half) {
    if (top_half) {
        return frame(Rect(0, 0, frame.cols, frame.rows / 2));
    }
    else {
        return frame(Rect(0, frame.rows / 2, frame.cols, frame.rows / 2));
    }
}

tuple<Mat, int, int, int> process_eye_crop(const Mat& frame, const vector<Rect>& eyes) {
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

tuple<vector<Mat>, vector<Mat>, vector<Mat>, vector<RotatedRect>>
generate_ellipse_candidates(const Mat& eye_gray, double dark_val) {
    vector<Mat> thresholded_images = threshold_images(eye_gray, dark_val);
    auto [contours, contour_images] = get_contours(thresholded_images);

    vector<Mat> ellipse_images;
    vector<RotatedRect> ellipses;

    for (const auto& cnt_list : contours) {
        Mat temp_img = eye_gray.clone();

        if (cnt_list.empty()) {
            ellipses.push_back(RotatedRect());
            ellipse_images.push_back(temp_img);
            continue;
        }

        RotatedRect box = fit_ellipse_custom(cnt_list);
        if (box.size.width > 0 && box.size.height > 0) {
            ellipse(temp_img, box, Scalar(255), 2);
            ellipses.push_back(box);
        }
        else {
            ellipses.push_back(RotatedRect());
        }
        ellipse_images.push_back(temp_img);
    }

    return make_tuple(thresholded_images, contour_images, ellipse_images, ellipses);
}

vector<double> calculate_ellipse_scores(const vector<Mat>& thresholded_images,
    const vector<RotatedRect>& ellipses) {
    vector<double> percents;

    for (size_t i = 0; i < thresholded_images.size(); i++) {
        const Mat& eye_thresh = thresholded_images[i];
        const RotatedRect& ellipse_rect = ellipses[i];

        if (ellipse_rect.size.width == 0 || ellipse_rect.size.height == 0) {
            percents.push_back(0);
            continue;
        }

        double ellipse_ratio = ellipse_rect.size.height / ellipse_rect.size.width;
        if (ellipse_ratio > 1.75 || ellipse_ratio < 0.8) {
            percents.push_back(0);
            continue;
        }

        Mat mask = Mat::zeros(eye_thresh.size(), CV_8UC1);
        ellipse(mask, ellipse_rect, Scalar(255), -1);

        int inside_total = countNonZero(mask);
        Mat inside_and;
        bitwise_and(eye_thresh, mask, inside_and);
        int inside_white = countNonZero(inside_and);
        double inside_ratio = inside_total > 0 ? (double)inside_white / inside_total : 0;

        Mat outside_mask;
        bitwise_not(mask, outside_mask);
        int outside_total = countNonZero(outside_mask);

        Mat eye_thresh_inv;
        bitwise_not(eye_thresh, eye_thresh_inv);
        Mat outside_and;
        bitwise_and(eye_thresh_inv, outside_mask, outside_and);
        int outside_black = countNonZero(outside_and);
        double outside_ratio = outside_total > 0 ? (double)outside_black / outside_total : 0;

        double percent = (inside_ratio + outside_ratio * 0.25) / 1.5;
        double w = ellipse_rect.size.width;
        double h = ellipse_rect.size.height;
        double roundness = 1.0 - abs(w - h) / max(w, h);

        percents.push_back(percent + roundness * 0);
    }

    return percents;
}

tuple<RotatedRect, int, int> select_best_ellipse(const vector<RotatedRect>& ellipses,
    const vector<double>& percents,
    int x, int y, int frame_idx) {
    auto max_it = max_element(percents.begin(), percents.end());
    size_t best_idx = distance(percents.begin(), max_it);
    RotatedRect best_ellipse = ellipses[best_idx];

    if (best_ellipse.size.width == 0 || best_ellipse.size.height == 0) {
        if (has_prev_ellipse) {
            cout << "Using previous ellipse" << endl;
            best_ellipse = prev_ellipse.ellipse;
            x = prev_ellipse.x;
            y = prev_ellipse.y;
        }
        else {
            cout << "No valid ellipse yet" << endl;
            return make_tuple(RotatedRect(), x, y);
        }
    }

    if (has_prev_ellipse) {
        Point2f pcenter = prev_ellipse.ellipse.center;
        Point2f center = best_ellipse.center;
        Size2f psize = prev_ellipse.ellipse.size;
        Size2f size = best_ellipse.size;

        if (abs(center.y - pcenter.y) > 100 || abs(center.x - pcenter.x) > 100) {
            cout << "Teleporting detected, using previous ellipse " << frame_idx << endl;
            best_ellipse = prev_ellipse.ellipse;
            x = prev_ellipse.x;
            y = prev_ellipse.y;
        }
        else if ((size.width * size.height) < 0.3 * (psize.width * psize.height)) {
            cout << "Current ellipse too small, using previous ellipse " << frame_idx << endl;
            best_ellipse = prev_ellipse.ellipse;
            x = prev_ellipse.x;
            y = prev_ellipse.y;
        }
    }

    return make_tuple(best_ellipse, x, y);
}

RotatedRect apply_smoothing(const RotatedRect& best_ellipse, float x, float y, RotatedRect ema, float x_alpha, float y_alpha, float width_alpha, float height_alpha, float rotation_alpha) {
    RotatedRect flipped = check_flip(best_ellipse);
    // check if we have ema
    if (ema.size.width == 0 || ema.size.height == 0) {
        return flipped;
    }
    Mat current = (Mat_<float>(5, 1) <<
        flipped.center.x,
        flipped.center.y,
        flipped.size.width,
        flipped.size.height,
        flipped.angle);

    Mat emalst = (Mat_<float>(5, 1) <<
        ema.center.x,
        ema.center.y,
        ema.size.width,
        ema.size.height,
		ema.angle);
    Mat alphas = (Mat_<float>(5, 1) <<
        x_alpha, y_alpha, width_alpha, height_alpha, rotation_alpha);

    emalst = alphas.mul(current) + (1.0 - alphas).mul(emalst);

    ema = RotatedRect(
        Point2f(emalst.at<float>(0), emalst.at<float>(1)),
        Size2f(emalst.at<float>(2), emalst.at<float>(3)),
        emalst.at<float>(4)
	);

    return ema;
}

void display_results(Mat& frame, const vector<Mat>& thresholded_images,
    const vector<Mat>& contour_images, const vector<Mat>& ellipse_images,
    const RotatedRect& full_ellipse, const Point2f& center,
    int x, int y, int frame_idx) {

    size_t N = thresholded_images.size();
    int H = thresholded_images[0].rows;
    int W = thresholded_images[0].cols;

    Mat grid = Mat::zeros(
        static_cast<int>(3 * H),
        static_cast<int>(N * W),
        CV_8UC1
    );

    for (int i = 0; i < N; i++) {
        Rect roi1(i * W, 0, W, H);
        Rect roi2(i * W, H, W, H);
        Rect roi3(i * W, 2 * H, W, H);

        thresholded_images[i].copyTo(grid(roi1));
        contour_images[i].copyTo(grid(roi2));
        ellipse_images[i].copyTo(grid(roi3));
    }

    Mat grid_disp;
    resize(grid, grid_disp, Size(1024, 512));
    imshow("Threshold | Contour | Ellipse", grid_disp);

    ellipse(frame, full_ellipse, Scalar(0, 255, 0), 2);
    circle(
        frame,
        Point(static_cast<int>(center.x + x), static_cast<int>(center.y + y)), 3, Scalar(0, 0, 255), -1);

    putText(frame, "Frame: " + to_string(frame_idx), Point(10, 30),
        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

    imshow("Eye Tracking", frame);
}
