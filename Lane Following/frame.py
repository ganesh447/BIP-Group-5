import cv2
import os
import glob

# --- CONFIGURATION ---
VIDEO_DIR = 'training_videos/'  # Folder containing your videos
OUTPUT_DIR = 'dataset_all/'     # Where frames will be saved
SKIP_FRAMES = 7                # Higher = more unique frames, Lower = more total frames
TARGET_SIZE = (224, 224)        # JetBot standard size

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Get all video files
extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
video_files = []
for ext in extensions:
    video_files.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))

print(f"Found {len(video_files)} videos. Starting extraction...")

total_saved = 0

for video_path in video_files:
    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    saved_from_this_video = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % SKIP_FRAMES == 0:
            # Resize immediately to save disk space
            frame = cv2.resize(frame, TARGET_SIZE)
            
            # Name file with video source + index to prevent overwriting
            filename = f"{video_name}_f{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)
            saved_from_this_video += 1
            total_saved += 1
            
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {saved_from_this_video} frames from {video_name}")

print(f"\n--- SUCCESS ---")
print(f"Total frames in {OUTPUT_DIR}: {total_saved}")