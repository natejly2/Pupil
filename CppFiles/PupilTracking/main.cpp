// main.cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include "npu_util.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "tracking_methods.h"
#include <filesystem>
using namespace npu_util;
using namespace cv;
using namespace Ort;
using namespace std;

struct EyeCropResult {
    cv::Mat gray;   
    int x, y, size;
};
int main(int argc, char** argv) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "eye-mask");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetInterOpNumThreads(1);
    opts.SetExecutionMode(ORT_SEQUENTIAL);
    auto session_options = Ort::SessionOptions();

    //auto cache_dir = std::filesystem::current_path().string();
    //auto npu_info = checkCompatibility_RAI_1_5();
    //std::cout << " - NPU Device ID     : 0x" << std::hex << npu_info.device_id << std::dec << std::endl;
    //std::cout << " - NPU Device Name   : " << npu_info.device_name << std::endl;
    //std::cout << " - NPU Driver Version: " << npu_info.driver_version_string << std::endl;

    //std::unordered_map<std::string, std::string> vai_ep_options;
  //  vai_ep_options["config_file"] = "C:/Users/VETi/source/repos/PupilTracking/x64/Release/vitisai_config.json";
  //  vai_ep_options["cache_dir"] = "my_cache_dir";
  //  vai_ep_options["cache_key"] = "keyidk";
  //  vai_ep_options["xclbin"] = "C:/Program Files/RyzenAI/1.5.0/voe - 4.0 - win_amd64/xclbins/phoenix/4x4.xclbin";
  //  try {
  //      session_options.AppendExecutionProvider_VitisAI(vai_ep_options);
		//printf("Execution Provider VitisAI appended successfully.\n");
  //  }
  //  catch (const std::exception& e) {
  //      std::cerr << "Exception occurred in appending execution provider: " << e.what() << std::endl;
  //  }
    //auto all_providers = Ort::GetAvailableProviders();
    //for (auto& p : all_providers) {
    //    std::cout << p << "\n";
    //}
    const ORTCHAR_T* model_path = ORT_TSTR("C:/Desktop/Pupil/model.onnx");
    Session session(env, model_path, session_options);

    AllocatorWithDefaultOptions allocator;
    AllocatedStringPtr inputNamePtr = session.GetInputNameAllocated(0, allocator);
    AllocatedStringPtr outputNamePtr = session.GetOutputNameAllocated(0, allocator);

    const char* inputName = inputNamePtr.get();
    const char* outputName = outputNamePtr.get();

    cout << "ONNX model loaded." << endl;

    cv::VideoCapture cap("C:/Desktop/Pupil/testvids/nate1.mp4");
    if (!cap.isOpened()) {
        cerr << "Cannot open video" << endl;
        return -1;
    }

    // smoothing parameters
    const float x_alpha = 0.85;
    const float y_alpha = 0.75;
    const float width_alpha = 0.5;
    const float height_alpha = 0.5;
    const float rotation_alpha = 1.0;
    bool top_half = false;

    MemoryInfo memInfo = MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    cv::RotatedRect emaEllipse;
    vector<cv::Rect> prevEyes;
    cv::RotatedRect prevEllipse;
    vector<cv::Rect> eyes;
    CascadeClassifier eye_cascade;
    cv::Mat frame;
    cv::Mat resized_eye(128, 128, CV_8UC1);
    cv::Mat eye_float(128, 128, CV_32FC1);
    cv::Mat mask(128, 128, CV_8U);

    vector<vector<cv::Point>> contours;
    contours.reserve(10);  // Reserve space
    array<int64_t, 4> inputDims = { 1, 128, 128, 1 };
    vector<float> inputValues(1 * 128 * 128 * 1);

    auto start  = chrono::high_resolution_clock::now();

    int frameIdx = 0;
    while (true) {
        if (!cap.read(frame))
            break;

		frame = prepare_frame(frame, top_half);

        if (cv::mean(frame)[0] < 30.0f) {
            cv::putText(frame, "NO PERSON", { 50,50 },
            cv::FONT_HERSHEY_SIMPLEX, 1, { 0,0,255 }, 2);
            printf("NO PERSON\n");

			cv::imshow("original", frame);
            if (cv::waitKey(1) == 'q') break;
            frameIdx++;
			continue;
        }

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

        if (eyes.empty()) {
            cv::putText(frame, "No Eyes", { 50,50 },
                cv::FONT_HERSHEY_SIMPLEX, 1, { 0,0,255 }, 2);
            cv::imshow("original", frame);
            printf("NO EYES\n");

            if (cv::waitKey(1) == 'q') break;
            frameIdx++;
            continue;
        }
        auto [eye, x, y, sz] = process_eye_crop(frame, eyes);

        cv::resize(eye, resized_eye, {128, 128}, 0, 0, cv::INTER_LINEAR);
        resized_eye.convertTo(eye_float, CV_32F, 1.0 / 255.0f);

        // Prepare input tensor (NHWC)

        memcpy(inputValues.data(), eye_float.data,
            inputValues.size() * sizeof(float));


        Value inputTensor = Value::CreateTensor<float>(
            memInfo, eye_float.ptr<float>(), inputValues.size(),
            inputDims.data(), inputDims.size()
        );

        // Run inference
        auto outputTensors = session.Run(
            RunOptions{ nullptr },
            &inputName, &inputTensor, 1,
            &outputName, 1
        );

        // Extract mask (assumes output shape [1,128,128,1])
        float* outData = outputTensors.front().GetTensorMutableData<float>();
        cv::threshold(cv::Mat(128, 128, CV_32F, outData), mask, 0.5, 255, cv::THRESH_BINARY);
        mask.convertTo(mask, CV_8U);

        // Contour & ellipse fitting
        vector<vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


        if (!contours.empty()) {
            // pick largest
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
                cv::RotatedRect ellipse = fitEllipse(contours[maxIdx]);

                // Rescale back to original frame coords
                float scale = float(sz) / 128.f;
                ellipse.center.x = ellipse.center.x * scale + x;
                ellipse.center.y = ellipse.center.y * scale + y;
                ellipse.size.width *= scale;
                ellipse.size.height *= scale;
                 
                 
                RotatedRect smooth_ellipse = apply_smoothing(ellipse, static_cast<float>(x), static_cast<float>(y), prevEllipse, x_alpha, y_alpha, width_alpha, height_alpha, rotation_alpha);
                prevEllipse = smooth_ellipse;

                // Draw
                cv::ellipse(frame, smooth_ellipse, {0, 255, 0}, 2);
                cv::circle(frame,
                    { int(smooth_ellipse.center.x), int(smooth_ellipse.center.y) },
                    3, {0, 0, 255}, -1);
            }
        }
        else {
            cv::putText(frame, "No Eyes", {100, 300},
                cv::FONT_HERSHEY_SIMPLEX, 1, {0, 0, 255}, 2);
                printf("NO EYES\n");
        }

        cv::imshow("original", frame);
        if (cv::waitKey(1) == 'q') break;
        frameIdx++;

    }

    auto end = chrono::high_resolution_clock::now();
    double secs = chrono::duration<double>(end - start).count();

    cout << "Processed " << frameIdx
        << " frames in " << secs << "s ("
        << (frameIdx / secs) << " FPS)\n";
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
