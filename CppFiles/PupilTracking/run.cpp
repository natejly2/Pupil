#include "pupil_tracking.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

int main() {
    try {
        PupilTracker* tracker = new PupilTracker("C:/Desktop/Pupil/model.onnx", 1);

        // Process a video file
        tracker->setSmoothingParameters(0.6f, 0.6f, 0.5f, 0.5f, 1.0f);
        tracker->processVideo("C:/Desktop/Pupil/PythonFiles/Media/TestVids/nate2.mp4");


    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}