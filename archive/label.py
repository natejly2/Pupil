import cv2
import numpy as np
import time
import os
import sys
import math
from math import pi
from tracking_methods import (coarse_find, remove_bright_spots, find_dark_area, threshold_images, 
                        get_contours, fit_ellipse, check_flip, prepare_frame, process_eye_crop, 
                        generate_ellipse_candidates, calculate_ellipse_scores, select_best_ellipse, 
                        apply_smoothing, display_results, check_blink)

if __name__ == "__main__":
    # get first arg

    frames_folder = "frames"
    x_alpha = .75
    y_alpha = .75
    rotation_alpha = 1
    width_alpha = .5
    height_alpha = .1
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
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("checks"):
        os.makedirs("checks")
    # if output/eye_data.csv exists, delete it
    if os.path.exists("output/eye_data.csv"):
        os.remove("output/eye_data.csv")
    with open("output/eye_data.csv", "w") as f:
        f.write("frame_idx,x,y,w,h,ellipse_x,ellipse_y,ellipse_w,ellipse_h,ellipse_angle\n")

    for i in range(num_frames):
        # status update
        if i % 100 == 0:
            print(f"Processing frame {i}/{num_frames}...")
        frame_path = os.path.join(frames_folder, f"{i}.png")
        if os.path.getsize(frame_path) < 20 * 1024:  # 20 KB
            print(f"New video detected at frame {i}, resetting EMA.")
            ema = None
            continue
        frame = cv2.imread(frame_path)
        eyes = coarse_find(frame)

        if len(eyes) > 0:
            prev_eyes = eyes.copy()
        elif prev_eyes is not None:
            eyes = prev_eyes
        else:
            continue

        eye_gray, x, y, size = process_eye_crop(frame, eyes)
        dark_square, dark_val = find_dark_area(eye_gray)
        to_save = eye_gray.copy()
        thresholded_images, contour_images, ellipse_images, ellipses = generate_ellipse_candidates(
            eye_gray, dark_val, thresholds)
        
        percents = calculate_ellipse_scores(thresholded_images, ellipses)
        
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

        if check_blink(frame, final_ellipse):
            continue
        cv2.ellipse(eye_gray, (int(cx-x), int(cy-y)), (int(w/2), int(h/2)), ang, 0, 360, (0, 255, 0), 2)
        
        cv2.imshow("eye_crop", eye_gray)
        frame_idx += 1
        prev_array.append(final_ellipse)
        cv2.imwrite(f"output/{frame_idx}.png", to_save)
        # save frame index, eye coordinates, and ellipse parameters
        cv2.imwrite(f"checks/eye_{frame_idx}.png", eye_gray)

        with open("output/eye_data.csv", "a") as f:
            f.write(f"{frame_idx},{x},{y},{w},{h},"
                    f"{final_ellipse[0][0]-x},{final_ellipse[0][1]-y},"
                        f"{final_ellipse[1][0]},{final_ellipse[1][1]},{final_ellipse[2]}\n")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # save video
    end_time = time.time()
    print(f"FPS: {frame_idx / (end_time - start_time):.2f}")
    cv2.destroyAllWindows()