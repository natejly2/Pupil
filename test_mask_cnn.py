import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import time
import numpy as np
import tensorflow as tf
from tracking_methods import coarse_find, process_eye_crop, fit_ellipse

interpreter = tf.lite.Interpreter(model_path="mask_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("TFLite models loaded.")

# Video capture
cap = cv2.VideoCapture("videos/2L.mp4")
if not cap.isOpened():
    raise IOError("Cannot open video")

top_half = False

ema = None
prev_eyes = None
frame_idx = 0
start_time = time.time()

# Frame processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop top or bottom half
    if top_half:
        frame = frame[:frame.shape[0] // 2, :]
    else:
        frame = frame[frame.shape[0] // 2:, :]
    if frame_idx % 10 == 0:  # Process every frame
        eyes = coarse_find(frame)
        if len(eyes) > 0:
            prev_eyes = eyes.copy()
        elif prev_eyes is not None:
            eyes = prev_eyes
        else:
            continue

    eye_gray, x, y, size = process_eye_crop(frame, eyes)
    # run inference to get mask
    eye_gray = cv2.resize(eye_gray, (128, 128))  #
    eye_gray = eye_gray / 255.0  # Normalize to [0, 1]
    eye_gray = eye_gray.reshape((1, 128, 128, 1))
    # convert to float32
    eye_gray = eye_gray.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], eye_gray)
    interpreter.invoke()
    pred_mask = interpreter.get_tensor(output_details[0]['index'])
    pred_mask = (pred_mask > 0.5).astype("uint8")
    pred_mask = pred_mask[0].squeeze()  # Remove batch dimension
    pred_mask = cv2.resize(pred_mask, (eye_gray.shape[2], eye_gray.shape[1]))
    # contours 
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    ellipse = fit_ellipse(contour) if contour is not None and len(contour) >= 5 else None
    # draw ellipse on the original frame
    # adjust ellipse coordinates and resize 
    scale = size / 128.0
    if ellipse is not None:
        (cx, cy), (w, h), ang = ellipse
        cx = cx * scale + x
        cy = cy * scale + y
        w *= scale
        h *= scale
        ellipse = ((cx, cy), (w, h), ang)
    cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
    cv2.circle(frame, (int(cx), int(cy)), 2, (0, 0, 255), -1)

    # overlay on frame
    cv2.imshow("original", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Processed {frame_idx} frames in {time.time() - start_time:.2f} seconds.")
print(f"FPS: {frame_idx / (time.time() - start_time):.2f}")