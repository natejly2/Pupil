import os
# (Optional) suppress other libraries' logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import time
import numpy as np
import onnxruntime as ort
from tracking_methods import coarse_find, process_eye_crop, fit_ellipse, apply_smoothing

# ——— LIME imports ———
# from lime import lime_image
from skimage.segmentation import mark_boundaries

# ——— Load ONNX model ———
session = ort.InferenceSession("PythonFiles\model7.onnx", providers=["VitisAIExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Registered providers:", ort.get_available_providers())

# ——— LIME helper: wrap segmentation -> 2-class prob fn ———
def segmentation_prob(rgb_images):
    """
    rgb_images: list or array of H×W×3 uint8 RGB images.
    Returns an array of shape (N,2) with [prob_no_pupil, prob_pupil].
    """
    out = []
    for img in rgb_images:
        # 1) to grayscale [0,1]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
        # 2) resize to model input
        gray_resized = cv2.resize(gray, (128,128)).astype(np.float32)
        inp = gray_resized.reshape(1,128,128,1)
        # 3) ONNX inference -> mask probabilities
        prob_mask = session.run([output_name], {input_name: inp})[0]  # shape (1,128,128,1)
        # 4) collapse to single prob: avg positive-class score
        p = float(prob_mask.mean())
        out.append([1 - p, p])
    return np.array(out)

# ——— Set up LIME explainer ———
# explainer = lime_image.LimeImageExplainer()

cap = cv2.VideoCapture(r"C:\Projects\pupil_detection\pupil_detection-main\PythonFiles\Media\videos\jg1.mp4")
if not cap.isOpened():
    raise IOError("Cannot open video")

# smoothing alphas
x_alpha, y_alpha = .75, .75
rotation_alpha = 1
width_alpha, height_alpha = .5, .5

top_half = False
ema = None
prev_eyes = None
prev_ellipse = None
frame_idx = 0
start_time = time.time()

lime_done = True  # flag so we only run LIME once

while True:
    ret, oframe = cap.read()
    if not ret:
        break
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Crop top/bottom
    h, w = oframe.shape[:2]
    frame = oframe[:h//2] if top_half else oframe[h//2:]
    avg_color = cv2.mean(frame)[:3]
    if avg_color < (50,50,50):
        cv2.putText(frame, "NO PERSON", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("original", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_idx += 1
        continue

    # Every 5th frame, detect eyes
    if frame_idx % 1 == 0:
        eyes = coarse_find(frame, min_size=(200,200))
        if len(eyes)>0:
            prev_eyes = eyes.copy()
        elif prev_eyes is not None:
            eyes = prev_eyes
        else:
            cv2.putText(frame, "No Eyes", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("original", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
            frame_idx += 1
            continue

    # Crop and preprocess eye patch
    eye_gray, x, y, size = process_eye_crop(frame, eyes)
    # run clahe on eye_gray

    eye_gray = cv2.resize(eye_gray, (128,128)) / 255.0
    eye_input = eye_gray.astype(np.float32).reshape(1,128,128,1)

    # ——— ONNX inference ———
    pred_mask = session.run([output_name], {input_name: eye_input})[0]
    pred_mask = (pred_mask>0.5).astype(np.uint8).squeeze()
    pred_mask = cv2.resize(pred_mask, (eye_input.shape[2], eye_input.shape[1]))
    cv2.imshow("pred_mask", pred_mask * 255)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # ——— LIME explanation (once) ———
    # if not lime_done:
    #     # convert grayscale to RGB for LIME
    #     img_rgb = (eye_gray*255).astype(np.uint8)
    #     img_rgb = np.stack([img_rgb]*3, axis=2)
    #     explanation = explainer.explain_instance(
    #         image=img_rgb,
    #         classifier_fn=segmentation_prob,
    #         top_labels=1,
    #         hide_color=0,
    #         num_samples=1000
    #     )
    #     temp, mask = explanation.get_image_and_mask(
    #         explanation.top_labels[0],
    #         positive_only=True,
    #         num_features=5,
    #         hide_rest=False
    #     )
    #     lime_overlay = mark_boundaries(temp, mask)
    #     # show it
    #     cv2.imshow("LIME Explanation", lime_overlay[...,::-1])  # RGB->BGR for OpenCV
    #     lime_done = True

    # Fit ellipse on mask
    contours,_ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    ellipse = fit_ellipse(contour) if contour is not None and len(contour)>=5 else None

    # Rescale & smooth & draw...
    scale = size/128.0
    if ellipse is not None:
        (cx, cy), (w_e, h_e), ang = ellipse
        cx, cy = cx*scale+x, cy*scale+y
        w_e, h_e = w_e*scale, h_e*scale
        ellipse = ( (cx,cy), (w_e,h_e), ang )
        ellipse, ema = apply_smoothing(
            ellipse, 0, 0, ema,
            x_alpha=x_alpha, y_alpha=y_alpha,
            width_alpha=width_alpha, height_alpha=height_alpha,
            rotation_alpha=rotation_alpha
        )
        prev_ellipse = ellipse
        cv2.ellipse(frame, ellipse, (0,255,0), 2)
        cv2.circle(frame, (int(cx),int(cy)), 3, (0,0,255), -1)
    else:
        cv2.putText(frame, "No Eyes", (100,300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("original", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
elapsed = time.time() - start_time
print(f"Processed {frame_idx} frames in {elapsed:.2f} seconds. FPS: {frame_idx/elapsed:.2f}")
cap.release()
cv2.destroyAllWindows()
