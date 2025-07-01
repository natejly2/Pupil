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

class EyeTracker {
private:
    CascadeClassifier eye_cascade;
    Mat mask;
    vector<Rect> prev_eyes;
    EllipseData prev_ellipse;
    bool has_prev_ellipse;
    Mat ema;
    bool has_ema;
    vector<int> thresholds;
    
    // Smoothing parameters
    float center_alpha;
    float size_alpha;
    float rotation_alpha;
    bool TOP;

public:
    EyeTracker() : has_prev_ellipse(false), has_ema(false), 
                   center_alpha(0.9f), size_alpha(0.25f), rotation_alpha(1.0f), TOP(false) {
        
        // Initialize cascade classifier
        if (!eye_cascade.load(samples::findFile("haarcascade_eye.xml"))) {
            cerr << "Error loading eye cascade!" << endl;
            exit(-1);
        }
        
        // Initialize mask
        mask = Mat::zeros(128, 128, CV_8UC1);
        
        // Initialize thresholds
        thresholds = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48};
    }
    
    vector<Rect> coarse_find(const Mat& frame) {
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        vector<Rect> eyes;
        eye_cascade.detectMultiScale(gray, eyes, 1.05, 3, 0, Size(150, 150));
        
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
                kept = {hull};
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
        } else if (ang < -90) {
            ang += 180;
        }
        
        ellipse.size.width = w;
        ellipse.size.height = h;
        ellipse.angle = ang;
        
        return ellipse;
    }
    
    Mat prepare_frame(const Mat& frame) {
        if (TOP) {
            return frame(Rect(0, 0, frame.cols, frame.rows / 2));
        } else {
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
            } else {
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
        int best_idx = distance(percents.begin(), max_it);
        RotatedRect best_ellipse = ellipses[best_idx];
        
        if (best_ellipse.size.width == 0 || best_ellipse.size.height == 0) {
            if (has_prev_ellipse) {
                cout << "Using previous ellipse" << endl;
                best_ellipse = prev_ellipse.ellipse;
                x = prev_ellipse.x;
                y = prev_ellipse.y;
            } else {
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
            } else if ((size.width * size.height) < 0.3 * (psize.width * psize.height)) {
                cout << "Current ellipse too small, using previous ellipse " << frame_idx << endl;
                best_ellipse = prev_ellipse.ellipse;
                x = prev_ellipse.x;
                y = prev_ellipse.y;
            }
        }
        
        return make_tuple(best_ellipse, x, y);
    }
    
    RotatedRect apply_smoothing(const RotatedRect& best_ellipse, int x, int y) {
        RotatedRect flipped = check_flip(best_ellipse);
        
        Mat current = (Mat_<float>(5, 1) << 
            flipped.center.x + x, 
            flipped.center.y + y, 
            flipped.size.width, 
            flipped.size.height, 
            flipped.angle);
        
        Mat alphas = (Mat_<float>(5, 1) << 
            center_alpha, center_alpha, size_alpha, size_alpha, rotation_alpha);
        
        if (!has_ema) {
            ema = current.clone();
            has_ema = true;
        } else {
            ema = alphas.mul(current) + (1.0 - alphas).mul(ema);
        }
        
        RotatedRect full_ellipse;
        full_ellipse.center.x = ema.at<float>(0);
        full_ellipse.center.y = ema.at<float>(1);
        full_ellipse.size.width = ema.at<float>(2);
        full_ellipse.size.height = ema.at<float>(3);
        full_ellipse.angle = ema.at<float>(4);
        
        return full_ellipse;
    }
    
    void display_results(Mat& frame, const vector<Mat>& thresholded_images, 
                        const vector<Mat>& contour_images, const vector<Mat>& ellipse_images,
                        const RotatedRect& full_ellipse, const Point2f& center, 
                        int x, int y, int frame_idx) {
        
        int N = thresholded_images.size();
        int H = thresholded_images[0].rows;
        int W = thresholded_images[0].cols;
        
        Mat grid = Mat::zeros(3 * H, N * W, CV_8UC1);
        
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
        circle(frame, Point(center.x + x, center.y + y), 3, Scalar(0, 0, 255), -1);
        
        putText(frame, "Frame: " + to_string(frame_idx), Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        
        imshow("Eye Tracking", frame);
    }
    
    void run(const string& video_path) {
        VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            cerr << "Error opening video file: " << video_path << endl;
            return;
        }
        
        int frame_idx = 0;
        auto start_time = chrono::high_resolution_clock::now();
        
        while (true) {
            Mat frame;
            if (!cap.read(frame)) {
                break;
            }
            
            Mat processed_frame = prepare_frame(frame);
            vector<Rect> eyes = coarse_find(processed_frame);
            
            if (!eyes.empty()) {
                prev_eyes = eyes;
            } else if (!prev_eyes.empty()) {
                eyes = prev_eyes;
            } else {
                continue;
            }
            
            auto [eye_gray, x, y, size] = process_eye_crop(processed_frame, eyes);
            auto [dark_square, dark_val] = find_dark_area(eye_gray);
            
            auto [thresholded_images, contour_images, ellipse_images, ellipses] = 
                generate_ellipse_candidates(eye_gray, dark_val);
            
            vector<double> percents = calculate_ellipse_scores(thresholded_images, ellipses);
            
            auto [best_ellipse, new_x, new_y] = select_best_ellipse(ellipses, percents, x, y, frame_idx);
            
            if (best_ellipse.size.width == 0 || best_ellipse.size.height == 0) {
                continue;
            }
            
            prev_ellipse = {best_ellipse, new_x, new_y};
            has_prev_ellipse = true;
            
            RotatedRect full_ellipse = apply_smoothing(best_ellipse, new_x, new_y);
            
            display_results(processed_frame, thresholded_images, contour_images, ellipse_images,
                           full_ellipse, best_ellipse.center, new_x, new_y, frame_idx);
            
            frame_idx++;
            
            if (waitKey(1) == 'q') {
                break;
            }
        }
        
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        double seconds = duration.count() / 1000.0;
        
        cout << "Processed " << frame_idx << " frames in " << seconds << " seconds." << endl;
        cout << "Average FPS: " << frame_idx / seconds << endl;
        
        cap.release();
        destroyAllWindows();
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <video_path>" << endl;
        return -1;
    }
    
    EyeTracker tracker;
    tracker.run(argv[1]);
    return 0;
}