import os
import glob
import cv2

VIDEO_FOLDER  = "videos"
OUTPUT_FOLDER = "frames"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

video_paths = sorted(glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4")))
if not video_paths:
    raise RuntimeError(f"No .mp4 files found in {VIDEO_FOLDER!r}")

global_idx = 0

for vp in video_paths:
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        print(f"⚠️ Skipping unreadable file: {vp}")
        continue

    basename = os.path.splitext(os.path.basename(vp))[0]
    print(f"Processing {basename} …")

    top_frames = []
    bottom_frames = []

    first = True
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if first:
            h, w = frame.shape[:2]
            h2 = h // 2
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Create video writers for split videos

            first = False

        top = frame[0:h2, :]
        bottom = frame[h2:h, :]

        top_frames.append(top)
        bottom_frames.append(bottom)

    cap.release()

    # Save top frames first
    for frame in top_frames:
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{global_idx}.png"), frame)
        global_idx += 1
    # save blank png so we can reset ema for new vid
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{global_idx}.png"), frame * 0)
    global_idx += 1

    # Then save bottom frames
    for frame in bottom_frames:
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{global_idx}.png"), frame)
        global_idx += 1
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{global_idx}.png"), frame * 0)
    global_idx += 1
print(f"✅ Saved {global_idx} frames in '{OUTPUT_FOLDER}' folder.")