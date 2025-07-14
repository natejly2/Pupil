#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    cv::Mat image = cv::Mat::zeros(400, 400, CV_8UC3);

    cv::circle(image, cv::Point(200, 200), 100, cv::Scalar(255, 255, 255), -1);

    cv::imshow("Test Window", image);
    cv::waitKey(0); 

    return 0;
}
