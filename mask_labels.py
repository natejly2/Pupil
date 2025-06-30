import cv2
import numpy as np
import time
import os
import sys
import math
from math import pi
from tracking_algo import (coarse_find, remove_bright_spots, find_dark_area, threshold_images, 
                        get_contours, fit_ellipse, check_flip, prepare_frame, process_eye_crop, 
                        generate_ellipse_candidates, calculate_ellipse_scores, select_best_ellipse, 
                        apply_smoothing, display_results, check_blink)

if __name__ == "__main__":
    # get first arg

    frames_folder = "frames"
    x_alpha = 1
    y_alpha = 1
    rotation_alpha = 1
    width_alpha = 1
    height_alpha = 1
    prev_array = []
    prev_ellipse = None
    ema = None

    # thresholds=[0, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    # brute force this for now 
    thresholds = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    TOP = False
    frame_idx = 0
    prev_eyes = None
    start_time = time.time()
# loop through images in frames_folder

    if not os.path.exists("frames"):
        print("No frames found in 'frames' folder. Please run 'processVideo.py' first.")
        sys.exit(1)
    # number of frames in frames_folder
    num_frames = len([f for f in os.listdir(frames_folder) if f.endswith('.png')])
    if not os.path.exists("best_masks"):
        os.makedirs("best_masks")
    if not os.path.exists("images"):
        os.makedirs("images")
    for i in range(num_frames):
        # status update
        if i % 100 == 0:
            print(f"Processing frame {i}/{num_frames}...")
        frame_path = os.path.join(frames_folder, f"{i}.png")

        frame = cv2.imread(frame_path)
        eyes = coarse_find(frame)

        if len(eyes) > 0:
            prev_eyes = eyes.copy()
        elif prev_eyes is not None:
            eyes = prev_eyes
        else:
            continue

        eye_gray, x, y, size, spots = process_eye_crop(frame, eyes, draw_mask=True)
        dark_square, dark_val = find_dark_area(eye_gray)
        to_save = eye_gray.copy()
        thresholded_images, contour_images, ellipse_images, ellipses = generate_ellipse_candidates(
            eye_gray, dark_val, thresholds)
        
        percents = calculate_ellipse_scores(thresholded_images, ellipses)
        # get index of max percent
        best_idx = int(np.argmax(percents))
        best_threshold = thresholded_images[best_idx]
        best_ellipse, x, y, save = select_best_ellipse(ellipses, percents, prev_ellipse, x, y, frame_idx)

        if not save:
            continue

        if best_ellipse is None:
            continue
            
        prev_ellipse = (best_ellipse, x, y)
        
        final_ellipse, ema = apply_smoothing(best_ellipse, x, y, ema,x_alpha=x_alpha,
                                            y_alpha=y_alpha,
                                            width_alpha=width_alpha,
                                            height_alpha=height_alpha,  
                                            rotation_alpha=rotation_alpha) 
        (cx, cy), (w, h), ang = final_ellipse
        prev_array.append(final_ellipse)

        if check_blink(frame, final_ellipse):
            continue
        # convert best_threshold to color
        # best_threshold = cv2.cvtColor(best_threshold, cv2.COLOR_GRAY2BGR)
        crop_ellipse = (int(cx - x), int(cy - y), int(w/1.95), int(h/1.95))

        # 1) make a single-channel mask the same size as best_threshold
        mask = np.zeros(best_threshold.shape[:2], dtype=np.uint8)

        # 2) draw a filled white ellipse into that mask
        center = (int(cx-x), int(cy-y))
        axes   = (int(w/1.95), int(h/1.95))
        cv2.ellipse(mask, center, axes, ang, 0, 360, 255, -1)

        spots = ~spots
        best_threshold = best_threshold.copy()
        # best_threshold[ np.any(spots, axis=2) ] = 255
        best_threshold = cv2.bitwise_and(best_threshold, mask)
        contour = cv2.findContours(best_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # show the largest contour
        contour = max(contour[0], key=cv2.contourArea)
        convex = cv2.convexHull(contour)
        export = np.zeros_like(frame)
        # translate contour to original frame coordinates
        convex = convex + np.array([[x, y]])
        # cv2.drawContours(frame, [convex], -1, (255, 255, 255), 2)
        cv2.drawContours(export, [convex], -1, (255,255,255), -1)
        cv2.imshow("images", frame)
        cv2.imshow("export", export)
        cv2.imwrite(f"images/{i}.png", frame)
        cv2.imwrite(f"best_masks/{i}.png", export)
        # cv2.imwrite(f"frames/{i}.png", frame)
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # save video
    end_time = time.time()
    print(f"FPS: {frame_idx / (end_time - start_time):.2f}")
    cv2.destroyAllWindows()