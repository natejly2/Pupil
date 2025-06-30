import cv2
import numpy as np
import time
import os
import sys
import math
from math import pi
mask = np.zeros((128, 128), dtype=np.uint8)
def coarse_find(frame):
    """Uses Haar filters to crop pic in to the eye"""
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(
        gray,
        # these params seem to give a good tradeoff between speed and accuracy
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(150, 150)
    )
    return eyes

def remove_bright_spots(image, threshold=200, replace=0):
    """replaces any pixels above threshold val """
    mask = image < threshold
    image[~mask] = replace
    return image, mask

# TODO: implement this function
def remove_thresholded_shadows():
    pass
def find_dark_area(image):
    """grid search darkest area of pixels to find threshold val"""
    num_grids = 9
    h, w = image.shape[:2]
    grid_h = h // num_grids
    grid_w = w // num_grids 
    darkest_val = 255
    darkest_square = None
    for i in range(num_grids):
        for j in range(num_grids):
            grid = image[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            mean_val = np.mean(grid)
            if mean_val < darkest_val:
                darkest_val = mean_val
                darkest_square = (i*grid_h, j*grid_w, grid_h, grid_w)
    return darkest_square, darkest_val
    
def threshold_images(image, dark_point, thresholds=[0, 5, 10, 15, 20, 25, 30, 35, 40, 50]):
    """Makes image array of thresholded images based on dark point and adjustment array"""
    images = []
    h, w = image.shape
    denoised = cv2.GaussianBlur(image, (5, 5), 0)   
    kernel = np.ones((3, 3), np.uint8)

    for t in thresholds:
        _, binary = cv2.threshold(
            denoised,
            dark_point + t,
            255,
            cv2.THRESH_BINARY_INV
        )
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        mask = np.zeros((h + 2, w + 2), np.uint8)

        #fill holes
        flood = opened.copy()
        cv2.floodFill(flood, mask, (0, 0), 255)

        flood_inv = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(opened, flood_inv)
        # remove small noise
        # gaussian blur to smooth edges
        filled = cv2.GaussianBlur(filled, (5, 5), 2)
        images.append(filled)

    return images

def get_contours(images, min_area=1500, margin=3):
    """gets contours for thresholded images"""
    filtered_contours = []
    contour_images    = []

    for img in images:
        h, w = img.shape[:2]
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        kept = []
        for cnt in cnts:
            all_pts = cnt.reshape(-1, 2)
            if cv2.contourArea(cnt) < min_area:
                continue
            pts = cnt.reshape(-1, 2)
            # if within margin pixels of edge of image remove
            if (pts[:,0] < margin).any() or (pts[:,0] > w - margin).any() \
            or (pts[:,1] < margin).any() or (pts[:,1] > h - margin).any():
                continue
            kept.append(cnt)
        kept = sorted(kept, key=cv2.contourArea, reverse=True)[:1]
        if len(kept) > 0:
            all_pts = np.vstack([c.reshape(-1,2) for c in kept])
            all_pts = all_pts.reshape(-1,1,2).astype(np.int32)
        else:
            all_pts = np.array([], dtype=np.int32).reshape(-1,1,2)

        # convex hull because we know pupil shouldn't be concave 
        hull = cv2.convexHull(all_pts)

        ci = np.zeros_like(img)
        cv2.drawContours(ci, hull, -1, 255, 2)
        filtered_contours.append(kept)
        contour_images.append(ci)

    return filtered_contours, contour_images

def fit_ellipse(contour, bias_factor=3):
    "Fits ellipse to cloud point. Bias factor forges points below y mean to bias to bottom"
    pts = np.vstack([c.reshape(-1,2) for c in contour])
    
    mean_y     = np.mean(pts[:,1])
    bottom_pts = pts[pts[:,1] > mean_y]
    
    if bottom_pts.size and bias_factor > 0:
        weighted_pts = np.concatenate(
            [pts] + [bottom_pts]*bias_factor,
            axis=0
        )
    else:
        weighted_pts = pts
    
    weighted_pts = weighted_pts.reshape(-1,1,2).astype(np.int32)
    if len(weighted_pts) < 5:
        return None
    temp = cv2.fitEllipse(weighted_pts)
    if temp is None:
        return None
    (cx, cy), (w, h), ang = temp
    if w < h:
        reflected_pts = weighted_pts.copy()
        reflected_pts[:, 0, 1] = 2 * cy - reflected_pts[:, 0, 1]
        weighted_pts = np.concatenate([weighted_pts, reflected_pts], axis=0)

    return cv2.fitEllipse(weighted_pts)

def check_flip(ellipse):
    "Normalizes angles and axes so width is always major axis"
    (cx, cy), (w, h), ang = ellipse


    if ang > 90:
        ang -= 180
    elif ang <= -90:
        ang += 180

    if ang == 90:
        ang = 0
        w, h = h, w
    return (cx, cy), (w, h), ang

def prepare_frame(frame, top_half=False):
    "Splits frame in half"
    if top_half:
        return frame[:frame.shape[0] // 2, :]
    else:
        return frame[frame.shape[0] // 2:, :]

def process_eye_crop(frame, eyes, draw_mask=False):
    "Function that crops the eye region"
    x, y, w, h = eyes[0]
    size = max(w, h)
    eye_crop = frame[y:y+size, x:x+size].copy()
    eye_crop, mask = remove_bright_spots(eye_crop, threshold=220, replace=100)
    eye_gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    if mask is None:
        mask = np.zeros_like(eye_gray, dtype=np.uint8)
    if not draw_mask:
        return eye_gray, x, y, size
    return eye_gray, x, y, size, mask

def generate_ellipse_candidates(eye_gray, dark_val, thresholds):
    "Function that makes the possible ellipses"
    thresholded_images = threshold_images(eye_gray, dark_val, thresholds=thresholds)
    contours, contour_images = get_contours(thresholded_images)
    ellipse_images = []
    ellipses = []
    
    for cnt_list in contours:
        temp_img = eye_gray.copy()

        if len(cnt_list) == 0:
            ellipses.append(None)
            ellipse_images.append(temp_img)
            continue
        
        box = fit_ellipse(cnt_list)
        cv2.ellipse(temp_img, box, 255, 2)
        ellipses.append(box)
        ellipse_images.append(temp_img)
    
    return thresholded_images, contour_images, ellipse_images, ellipses

def calculate_ellipse_scores(thresholded_images, ellipses):
    "Scores each ellipse by filtering axis ratios, size, and the ratio of white/black pixels on inside of elipse and vice versa for outside"
    N = len(thresholded_images) 
    percents = []
    
    for i in range(N):
        eye_thresh = thresholded_images[i]
        ellipse = ellipses[i]
        if ellipse is None:
            percents.append(0)
            continue
        
        ellipse_ratio = ellipse[1][1] / ellipse[1][0]
        if ellipse_ratio > 1.75 or ellipse_ratio < 0.8:
            percents.append(0)
            continue

        mask = np.zeros_like(eye_thresh)
        (cx, cy), (w, h), ang = ellipse
        cv2.ellipse(mask,
                    (int(cx), int(cy)),
                    (int(w/2), int(h/2)),
                    ang, 0, 360,
                    255, -1)

        inside_total  = cv2.countNonZero(mask)
        inside_white  = cv2.countNonZero(cv2.bitwise_and(eye_thresh, mask))
        inside_ratio  = inside_white / inside_total if inside_total>0 else 0

        outside_mask  = cv2.bitwise_not(mask)
        outside_total = cv2.countNonZero(outside_mask)
        outside_black = cv2.countNonZero(cv2.bitwise_and(cv2.bitwise_not(eye_thresh), outside_mask))
        outside_ratio = outside_black / outside_total if outside_total>0 else 0

        # some heuristic stuff here seems to work pretty well here 
        percent = ((inside_ratio + outside_ratio*.25) / 1.5)
        roundness = 1.0 - abs(w - h) / max(w, h)
        percents.append(percent + roundness*0)
    
    return percents

def select_best_ellipse(ellipses, percents, prev_ellipse, x, y, frame_idx, debug=False, save=True):
    "More elipse filtering and uses the previous ellipse if we violate conditions"
    best_idx = int(np.argmax(percents))
    best_ellipse = ellipses[best_idx]

    if best_ellipse is None:
        if prev_ellipse is not None:
            if debug:
                print(f"Using previous ellipse {frame_idx}")
            best_ellipse, prev_x, prev_y = prev_ellipse
            x, y = prev_x, prev_y
            save = False
        else:
            print("No valid ellipse yet")
            return None, x, y
    
    if prev_ellipse is not None:
        (pcx, pcy), (pw, ph), pang = prev_ellipse[0]
        (cx, cy), (w, h), ang = best_ellipse
        # Center moves too much
        if abs(cy - pcy) > 100 or abs(cx - pcx) > 100:
            if debug:
                print(f"Teleporting detected, using previous ellipse {frame_idx}")
            best_ellipse = prev_ellipse[0]
            x, y = prev_ellipse[1], prev_ellipse[2]
            save = False


        # Too small
        elif (w * h) < 0.3 * (pw * ph):
            if debug:
                print(f"Current ellipse too small, using previous ellipse {frame_idx}")
            best_ellipse = prev_ellipse[0]
            x, y = prev_ellipse[1], prev_ellipse[2]
            save = False

    return best_ellipse, x, y, save

def apply_smoothing(best_ellipse, x, y, ema, 
                    x_alpha, y_alpha, width_alpha, height_alpha, rotation_alpha):

    best_ellipse = check_flip(best_ellipse)
    (cx, cy), (w, h), ang = best_ellipse

    current = np.array([cx + x, cy + y, w, h, ang], dtype=np.float32)
    if ema is None:
        return best_ellipse, current.copy()
    alphas = np.array([x_alpha, y_alpha, width_alpha, height_alpha, rotation_alpha], dtype=np.float32)

    ema = alphas * current + (1.0 - alphas) * ema

    sm_cx, sm_cy, sm_w, sm_h, sm_ang = ema
    full_ellipse = (
        (float(sm_cx), float(sm_cy)),
        (float(sm_w), float(sm_h)),
        float(sm_ang)
    )

    return full_ellipse, ema

def check_blink(frame,
                full_ellipse,
                offset_x=0.0,
                offset_y=0.0,
                brightness_thresh=110.0):

    (cx, cy), (w, h), ang = full_ellipse

    center = (int(cx + offset_x), int(cy + offset_y))
    axes   = (int(w/2), int(h/2))

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, axes, ang, 0, 360, 255, -1)

    mean_bgr = cv2.mean(frame, mask=mask)[:3]
    mean_intensity = sum(mean_bgr) / 3.0

    # 5) Blink = region too bright
    return mean_intensity > brightness_thresh

def display_results(frame, thresholded_images, contour_images, ellipse_images, 
                    full_ellipse,cx, cy, x, y, frame_idx):
    "Shows frames"
    N = len(thresholded_images)
    H, W = thresholded_images[0].shape
    grid = np.zeros((3 * H, N * W), dtype=np.uint8)
    
    for i in range(N):
        grid[0   :  H, i*W:(i+1)*W] = thresholded_images[i]
        grid[H   :2*H, i*W:(i+1)*W] = contour_images[i]
        grid[2*H :3*H, i*W:(i+1)*W] = ellipse_images[i]
    
    grid_disp = cv2.resize(grid, (1024, 512))
    cv2.imshow("Threshold | Contour | Ellipse", grid_disp)
    cv2.ellipse(frame, full_ellipse, (0, 255, 0), 2)
    cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (255, 255, 255), 2)
    cv2.imshow("Eye Tracking", frame)

def main():

    video_path = "videos/igor1.mp4"
    TOP = True
    debug = True
    x_alpha = .75
    y_alpha = .75

    rotation_alpha = 1
    width_alpha = .5
    height_alpha = .1
    prev_array = []
    prev_ellipse = None
    ema = None
    # is this too many? Runs 30+ fps no problem
    thresholds = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    prev_eyes = None
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = prepare_frame(frame, TOP)
        if frame_idx % 5 == 0:
            eyes = coarse_find(frame)

            if len(eyes) > 0:
                prev_eyes = eyes.copy()
            elif prev_eyes is not None:
                eyes = prev_eyes
            else:
                continue

        eye_gray, x, y, size = process_eye_crop(frame, eyes)
        dark_square, dark_val = find_dark_area(eye_gray)
        
        thresholded_images, contour_images, ellipse_images, ellipses = generate_ellipse_candidates(
            eye_gray, dark_val, thresholds)
        
        percents = calculate_ellipse_scores(thresholded_images, ellipses)
        
        best_ellipse, x, y, _ = select_best_ellipse(ellipses, percents, prev_ellipse, x, y, frame_idx, debug=debug)
        
        if best_ellipse is None:
            continue
            
        prev_ellipse = (best_ellipse, x, y)

        full_ellipse, ema = apply_smoothing(best_ellipse, x, y, ema,x_alpha=x_alpha,
                                            y_alpha=y_alpha,
                                            width_alpha=width_alpha,
                                            height_alpha=height_alpha,  
                                            rotation_alpha=rotation_alpha) 
        
        (cx, cy), (w, h), ang = full_ellipse
        if check_blink(frame, full_ellipse):
                print(f"Blink detected at frame {frame_idx}")
                cv2.putText(frame, "Blink Detected", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        display_results(frame, thresholded_images, contour_images, ellipse_images, full_ellipse,
                       cx, cy, x, y, frame_idx)
        frame_idx += 1
        prev_array.append(full_ellipse)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    end_time = time.time()
    print(f"Processed {frame_idx} frames in {end_time - start_time:.2f} seconds.")
    print(f"Average FPS: {frame_idx / (end_time - start_time):.2f}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()