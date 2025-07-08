import os
# (Optional) suppress other libraries' logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import time
import numpy as np
import onnxruntime as ort
from tracking_methods import coarse_find, process_eye_crop, fit_ellipse, apply_smoothing, remove_bright_spots


# ——— Load ONNX model ———
session = ort.InferenceSession("model.onnx", providers=["VitisAIExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Registered providers:", ort.get_available_providers())

cap = cv2.VideoCapture("videos/nate1.mp4")
if not cap.isOpened():
    raise IOError("Cannot open video")

# smoothing alphas
x_alpha, y_alpha = .75, .75
rotation_alpha = 1
width_alpha, height_alpha = .5, .5

top_half = True
ema = None
prev_eyes = None
prev_ellipse = None
frame_idx = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not ret:
        break

    # Crop top or bottom half
    h, w = frame.shape[:2]
    frame = frame[:h//2] if top_half else frame[h//2:]  
    # get average color of frame
    avg_color = cv2.mean(frame)[:3]
    if avg_color < (50, 50, 50):
        cv2.putText(frame, "NO PERSON", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Every 5th frame, detect eyes
    if frame_idx % 5 == 0:
        eyes = coarse_find(frame, min_size=(200, 200))
        if len(eyes) > 0:
            prev_eyes = eyes.copy()
        elif prev_eyes is not None:
            eyes = prev_eyes
        else:
            cv2.putText(frame, "No Eyes", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("original", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_idx += 1
            continue

    # Crop and preprocess eye patch
    eye_gray, x, y, size = process_eye_crop(frame, eyes)
    eye_gray = cv2.resize(eye_gray, (128, 128)) / 255.0
    eye_input = eye_gray.astype(np.float32).reshape(1, 128, 128, 1)

    # ——— ONNX inference ———
    pred_mask = session.run([output_name], {input_name: eye_input})[0]
    # pred_mask shape: (1, 128,128,1) or (1,128,128) depending on your model
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    pred_mask = pred_mask.squeeze()  # -> (128,128)
    pred_mask = cv2.resize(pred_mask, (eye_input.shape[2], eye_input.shape[1]))

    # Find largest contour and fit ellipse
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    ellipse = fit_ellipse(contour) if contour is not None and len(contour) >= 5 else None

    # Rescale & smooth
    scale = size / 128.0
    if ellipse is not None:
        (cx, cy), (w_e, h_e), ang = ellipse
        cx = cx * scale + x
        cy = cy * scale + y
        w_e *= scale; h_e *= scale
        ellipse = (cx, cy), (w_e, h_e), ang

        ellipse, ema = apply_smoothing(
            ellipse, 0, 0, ema,
            x_alpha=x_alpha, y_alpha=y_alpha,
            width_alpha=width_alpha, height_alpha=height_alpha,
            rotation_alpha=rotation_alpha
        )
        prev_ellipse = ellipse

        # Draw
        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    else:
        cv2.putText(frame, "No Eyes", (100, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show and advance
    cv2.imshow("original", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
elapsed = time.time() - start_time
print(f"Processed {frame_idx} frames in {elapsed:.2f} seconds.")
print(f"FPS: {frame_idx / elapsed:.2f}")
cap.release()
cv2.destroyAllWindows()
