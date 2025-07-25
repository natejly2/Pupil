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
#TODO: clean code up 
if __name__ == "__main__":
    # get first arg

    FRAMES_FOLDER = r"PythonFiles\Media\frames"
    MASKS_FOLDER = r"PythonFiles\Media\masks"
    IMAGE_FOLDER = r"PythonFiles\Media\images"
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
# loop through images in FRAMES_FOLDER

    if not os.path.exists(FRAMES_FOLDER):
        print("No frames found in 'frames' folder. Please run 'processVideo.py' first.")
        sys.exit(1)
    # number of frames in FRAMES_FOLDER
    num_frames = len([f for f in os.listdir(FRAMES_FOLDER) if f.endswith('.png')])
    if not os.path.exists(MASKS_FOLDER):
        os.makedirs(MASKS_FOLDER)
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
    for i in range(num_frames):
        # status update
        if i % 100 == 0:
            print(f"Processing frame {i}/{num_frames}...")
        frame_path = os.path.join(FRAMES_FOLDER, f"{i}.png")

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
        # no convex here because bright spots
        export = np.zeros_like(eye_gray)
        # cv2.drawContours(eye_gray, [convex], -1, (255, 255, 255), 2)
        cv2.drawContours(export, [contour], -1, 255, -1)
        #cv2.imshow("images", eye_gray)
        #cv2.imshow("export", export)
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", eye_gray)
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", export)
        # cv2.imwrite(f"frames/{i}.png", frame)
        frame_idx += 1

        # flip across y axis
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", cv2.flip(eye_gray, 1))
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", cv2.flip(export, 1))
        # cv2.imwrite(f"frames/{i}.png", frame)
        frame_idx += 1


        # brightness augment
        bright = np.ones_like(eye_gray) * np.random.randint(10, 70)
        brightup = cv2.add(eye_gray, bright)
        #cv2.imshow("brightup", brightup)
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", brightup)
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", export)
        frame_idx += 1

        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", cv2.flip(brightup, 1))
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", cv2.flip(export, 1))
        # cv2.imwrite(f"frames/{i}.png", frame)
        frame_idx += 1


        # subtract brightness augment
        bright = np.ones_like(eye_gray) * np.random.randint(10, 70)
        brightdown = cv2.subtract(eye_gray, bright)
        #cv2.imshow("brightdown", brightdown)
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", brightdown)
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", export)
        frame_idx += 1

        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", cv2.flip(brightdown, 1))
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", cv2.flip(export, 1))
        frame_idx += 1

        # sharpness augment
        sharpval = np.random.randint(5, 10)
        kernel = np.array([[0, -1, 0],
                           [-1, sharpval, -1],
                           [0, -1, 0]])
        sharp = cv2.filter2D(eye_gray, -1, kernel)
        #cv2.imshow("sharp", sharp)
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", sharp)
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", export)
        frame_idx += 1
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", cv2.flip(sharp, 1))
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", cv2.flip(export, 1))
        frame_idx += 1

        #decrease sharpness augment
        kernel = np.ones((5, 5), np.float32) / 25
        sharpdown = cv2.filter2D(eye_gray, -1, kernel)
        #cv2.imshow("sharpdown", sharpdown)
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", sharpdown)
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", export)
        frame_idx += 1

        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", cv2.flip(sharpdown, 1))
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", cv2.flip(export, 1))
        frame_idx += 1

        #contrast augment
        contrast = np.random.uniform(0.5, 1.5)
        eye_gray2 = cv2.convertScaleAbs(eye_gray, alpha=contrast, beta=0)
        #cv2.imshow("contrast", eye_gray)
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", eye_gray2)
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", export)
        frame_idx += 1

        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", cv2.flip(eye_gray2, 1))
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", cv2.flip(export, 1))
        frame_idx += 1

        # blur augment
        blurval = np.random.randint(1, 6) * 2 - 1
        blur = cv2.GaussianBlur(eye_gray, (blurval, blurval), 0)
        #cv2.imshow("blur", blur)
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", blur)
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", export)
        frame_idx += 1
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", cv2.flip(blur, 1))
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", cv2.flip(export, 1))
        frame_idx += 1

        # gamma augment
        gamma = np.random.uniform(0.5, 2.0)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        gamma_corrected = cv2.LUT(eye_gray, table)
        #cv2.imshow("gamma_corrected", gamma_corrected)
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", gamma_corrected)
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", export)
        frame_idx += 1
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", cv2.flip(gamma_corrected, 1))
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", cv2.flip(export, 1))
        frame_idx += 1


        # rotate augment
        angle = np.random.randint(-30, 30)
        h, w = eye_gray.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_gray = cv2.warpAffine(eye_gray, rotation_matrix, (w, h))
        rotated_mask = cv2.warpAffine(export, rotation_matrix, (w, h))
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", rotated_gray)
        #cv2.imshow("rotated", rotated_gray)
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", rotated_mask)
        frame_idx += 1

        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", cv2.flip(rotated_gray, 1))
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", cv2.flip(rotated_mask, 1))
        frame_idx += 1

        # stretch/compress augment
        stretch_factor = np.random.uniform(0.8, 1.2)
        stretched_gray = cv2.resize(eye_gray, None, fx=stretch_factor, fy=1.0)
        stretched_mask = cv2.resize(export, None, fx=stretch_factor, fy=1.0)
        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", stretched_gray)
        #cv2.imshow("stretched", stretched_gray)
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", stretched_mask)
        frame_idx += 1

        cv2.imwrite(f"{IMAGE_FOLDER}/{frame_idx}.png", cv2.flip(stretched_gray, 1))
        cv2.imwrite(f"{MASKS_FOLDER}/{frame_idx}.png", cv2.flip(stretched_mask, 1))
        frame_idx += 1
        




        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # save video
    end_time = time.time()
    cv2.destroyAllWindows()