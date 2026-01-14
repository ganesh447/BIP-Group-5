import torch
import torchvision
import cv2
import numpy as np
import time
from jetbot import Robot, Camera, bgr8_to_jpeg
import ipywidgets.widgets as widgets
from IPython.display import display
import threading
from queue import Queue, Empty
from rplidar import RPLidar  # Install via: pip install rplidar

# --- 1. CONFIGURATION (Updated) ---
model_path = 'fainl_model.pth'  # Assuming this is your final model path
speed_gain = 0.15  # Base speed
steering_gain = 0.12  # INCREASED: Higher value = sharper turns (was 0.04)
steering_kd = 0.04  # NEW: Prevents overshooting/wobbling
steering_bias = 0.0  # Adjust (e.g., 0.01) if robot naturally veers one way

# LIDAR Configuration
LIDAR_PORT = '/dev/ttyACM1'  # Check your port: ls /dev/ttyUSB*
OBSTACLE_THRESHOLD = 500  # mm - distance to consider as obstacle
FRONT_ANGLE_RANGE = 60  # degrees either side of front (0 deg = front)
AVOID_TURN_STRENGTH = 0.5  # How sharply to turn during avoidance
AVOID_DURATION = 1.0  # Seconds to turn before re-checking

# FSM States
STATE_FOLLOW_LANE = 'FOLLOW_LANE'
STATE_AVOID_LEFT = 'AVOID_LEFT'
STATE_AVOID_RIGHT = 'AVOID_RIGHT'
STATE_STOP = 'STOP'

# Globals for FSM and LIDAR
current_state = STATE_FOLLOW_LANE
lidar_queue = Queue()  # For non-blocking LIDAR data
last_min_distance = float('inf')  # Default safe
avoid_start_time = 0.0

# --- 2. MODEL SETUP ---
device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load(model_path))
model = model.to(device).eval().half()  # FP16 for speed

# Normalization constants
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

# --- 3. HARDWARE & UI ---
robot = Robot()
camera = Camera.instance(width=224, height=224)

# Widgets
image_widget = widgets.Image(format='jpeg', width=224, height=224)
stop_button = widgets.Button(description='STOP ROBOT', button_style='danger',
                             layout={'width': '100%', 'height': '100px'})

# Layout the UI (Camera on left, Stop on right)
ui_layout = widgets.HBox([image_widget, stop_button])
display(ui_layout)

# --- 4. PREPROCESSING ---
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float().permute(2, 0, 1).to(device).half() / 255.0
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# --- 5. LIDAR Thread (Runs in background to avoid blocking) ---
def lidar_thread():
    lidar = RPLidar(LIDAR_PORT)
    lidar.start_motor()
    try:
        for scan in lidar.iter_scans(max_buf_meas=500):  # Limit buffer to prevent overflow
            front_distances = []
            left_distances = []
            right_distances = []
            for quality, angle, distance in scan:
                if quality > 0:  # Valid measurement
                    if (360 - FRONT_ANGLE_RANGE/2) > angle > (FRONT_ANGLE_RANGE/2):  # Adjust for RPLIDAR angle (0 front, increasing CCW)
                        angle = (angle + 180) % 360 - 180  # Normalize to -180 to 180, 0 front
                    if abs(angle) < FRONT_ANGLE_RANGE / 2:
                        front_distances.append(distance)
                    elif -90 < angle < -FRONT_ANGLE_RANGE / 2:
                        right_distances.append(distance)  # Right side
                    elif FRONT_ANGLE_RANGE / 2 < angle < 90:
                        left_distances.append(distance)  # Left side
            
            min_front = min(front_distances) if front_distances else float('inf')
            min_left = min(left_distances) if left_distances else float('inf')
            min_right = min(right_distances) if right_distances else float('inf')
            
            lidar_queue.put((min_front, min_left, min_right))  # Put tuple in queue
    except Exception as e:
        print(f"LIDAR Error: {e}")
    finally:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()

# Start LIDAR thread
threading.Thread(target=lidar_thread, daemon=True).start()

# --- 6. THE EXECUTION FUNCTION (With FSM Logic) ---
def execute(change):
    global current_state, last_x, last_min_distance, avoid_start_time
    
    image = change['new']
    
    # Get LIDAR data non-blocking
    try:
        min_front, min_left, min_right = lidar_queue.get_nowait()
        last_min_distance = min_front
    except Empty:
        min_front = last_min_distance  # Use last known
        min_left = float('inf')
        min_right = float('inf')
    
    # Run Inference for lane following
    with torch.no_grad():
        output = model(preprocess(image)).flatten()
    
    # Current horizontal error (-1.0 to 1.0)
    x = float(output[0])
    
    # --- FSM Logic ---
    if current_state == STATE_FOLLOW_LANE:
        if min_front < OBSTACLE_THRESHOLD:
            # Obstacle detected - decide avoidance direction
            if min_left > min_right:
                current_state = STATE_AVOID_RIGHT  # Turn right if left is more open
            else:
                current_state = STATE_AVOID_LEFT   # Turn left
            avoid_start_time = time.time()
            print(f"Obstacle detected! Avoiding {'left' if current_state == STATE_AVOID_LEFT else 'right'}")
        else:
            # Normal PD control for lane following
            change_in_x = x - last_x
            steering = (x * steering_gain) + (change_in_x * steering_kd) + steering_bias
            last_x = x
            
            left_v = max(min(speed_gain + steering, 1.0), 0.0)
            right_v = max(min(speed_gain - steering, 1.0), 0.0)
            robot.set_motors(left_v, right_v)
    
    elif current_state in (STATE_AVOID_LEFT, STATE_AVOID_RIGHT):
        # Perform avoidance maneuver
        steering = AVOID_TURN_STRENGTH if current_state == STATE_AVOID_LEFT else -AVOID_TURN_STRENGTH
        left_v = max(min(speed_gain + steering, 1.0), 0.0)
        right_v = max(min(speed_gain - steering, 1.0), 0.0)
        robot.set_motors(left_v, right_v)
        
        # Check if avoidance duration passed and front clear
        if time.time() - avoid_start_time > AVOID_DURATION and min_front > OBSTACLE_THRESHOLD:
            current_state = STATE_FOLLOW_LANE
            print("Avoidance complete - resuming lane following")
    
    # If too close or emergency, stop
    if min_front < OBSTACLE_THRESHOLD / 2:
        current_state = STATE_STOP
        robot.stop()
        print("Critical obstacle - FULL STOP!")
    
    # Visualization (Draws a line showing steering direction + red if obstacle)
    px = int(112 + x * 112)
    color = (0, 0, 255) if min_front < OBSTACLE_THRESHOLD else (0, 255, 0)  # Red if obstacle
    cv2.line(image, (112, 224), (px, 112), color, 3)
    image_widget.value = bgr8_to_jpeg(image)

# --- 7. STOP BUTTON LOGIC ---
def stop_all(c):
    global current_state
    camera.unobserve_all()
    time.sleep(0.5)
    robot.stop()
    current_state = STATE_STOP
    print("Emergency Stop: Robot Halted and Camera Released.")

stop_button.on_click(stop_all)

# --- 8. START ---
camera.observe(execute, names='value')
print("Robot is LIVE with LIDAR FSM. Click the RED BUTTON to stop.")
print("Install rplidar lib if needed: pip install rplidar")
print("Tune OBSTACLE_THRESHOLD, FRONT_ANGLE_RANGE, AVOID_TURN_STRENGTH as needed.")
print("Check LIDAR port - may need sudo or permissions: sudo chmod 666 /dev/ttyUSB0")
