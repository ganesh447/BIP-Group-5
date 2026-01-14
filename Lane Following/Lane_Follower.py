import torch
import torchvision
import cv2
import numpy as np
import time
import threading
from rplidar import RPLidar
from jetbot import Robot, Camera, bgr8_to_jpeg
import ipywidgets.widgets as widgets
from IPython.display import display

# --- 1. CONFIGURATION ---
model_path = 'fainl_model.pth' 
speed_gain = 0.15
steering_gain = 0.12
steering_kd = 0.04

# LiDAR Parameters
STOP_DIST = 200.0    # Stop if object is 20cm away
AVOID_DIST = 550.0   # Start looking for obstacles at 55cm
SAFE_CORRIDOR = 150.0 # Width of the "danger zone" around the lane path

# --- 2. LIDAR BACKGROUND THREAD ---
class LidarScanner(threading.Thread):
    def __init__(self, port='/dev/ttyACM1'):
        threading.Thread.__init__(self)
        self.lidar = RPLidar(port)
        self.lidar.start_motor()
        self.avg_x_raw = 0.0 
        self.min_dist = 9999.0
        self.stop_signal = False
        self.daemon = True

    def run(self):
        while not self.stop_signal:
            try:
                for scan in self.lidar.iter_scans(max_buf_meas=500):
                    if self.stop_signal: break
                    
                    danger_x = []
                    self.min_dist = 9999.0
                    
                    for (_, angle, dist) in scan:
                        if 0 < dist < AVOID_DIST:
                            # Frontal cone check (315 to 45 degrees)
                            if (angle > 315 or angle < 45):
                                if dist < self.min_dist:
                                    self.min_dist = dist
                                
                                # Convert Polar to horizontal X offset
                                rad = np.deg2rad(angle)
                                x_coord = dist * np.sin(rad)
                                danger_x.append(x_coord)
                    
                    if len(danger_x) > 5:
                        self.avg_x_raw = sum(danger_x) / len(danger_x)
                    else:
                        self.avg_x_raw = 0.0
                        
            except Exception:
                self.lidar.clean_input()

    def stop(self):
        self.stop_signal = True
        self.lidar.stop_motor()
        self.lidar.disconnect()

# --- 3. MODEL & HARDWARE SETUP ---
device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load(model_path))
model = model.to(device).eval().half()

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

scanner = LidarScanner()
scanner.start()

robot = Robot()
camera = Camera.instance(width=224, height=224)

# UI
image_widget = widgets.Image(format='jpeg', width=224, height=224)
label_widget = widgets.Label(value="Initializing...")
stop_button = widgets.Button(description='STOP ROBOT', button_style='danger')
display(widgets.VBox([image_widget, label_widget, stop_button]))

# --- 4. PREPROCESSING ---
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float().permute(2, 0, 1).to(device).half() / 255.0
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# --- 5. EXECUTION LOOP ---
last_x = 0.0



def execute(change):
    global last_x
    image = change['new']
    
    # A. GET SENSOR DATA
    dist = scanner.min_dist
    obs_x = scanner.avg_x_raw
    
    # B. AI LANE PREDICTION
    with torch.no_grad():
        output = model(preprocess(image)).flatten()
    ai_x = float(output[0])
    
    # C. SAFETY/FSM LOGIC
    avoid_nudge = 0.0
    state = "FOLLOWING"
    
    if dist < STOP_DIST:
        robot.stop()
        label_widget.value = f"STOPPED: Obstacle {int(dist)}mm"
        return

    # Check if obstacle is blocking the AI's predicted path
    # We map ai_x (-1 to 1) to a scale comparable to LiDAR X
    path_in_lidar_units = ai_x * 200.0 
    
    if dist < AVOID_DIST:
        distance_to_path = obs_x - path_in_lidar_units
        
        if abs(distance_to_path) < SAFE_CORRIDOR:
            state = "AVOIDING"
            # Steer opposite to the obstacle relative to the path
            avoid_nudge = -0.4 if distance_to_path > 0 else 0.4

    # D. BLENDED STEERING
    change_in_x = ai_x - last_x
    ai_steering = (ai_x * steering_gain) + (change_in_x * steering_kd)
    total_steering = ai_steering + avoid_nudge
    last_x = ai_x
    
    # E. DRIVE
    l_v = max(min(speed_gain + total_steering, 1.0), 0.0)
    r_v = max(min(speed_gain - total_steering, 1.0), 0.0)
    robot.set_motors(l_v, r_v)

    # F. VISUALS
    label_widget.value = f"State: {state} | Dist: {int(dist)}mm | Nudge: {avoid_nudge}"
    px = int(112 + ai_x * 112)
    cv2.line(image, (112, 224), (px, 112), (0, 255, 0), 3) # Green = Path
    if avoid_nudge != 0:
        cv2.circle(image, (int(112 + (ai_x + avoid_nudge)*112), 80), 10, (0, 0, 255), -1) # Red Dot = Avoidance Goal
    image_widget.value = bgr8_to_jpeg(image)

# --- 6. SHUTDOWN ---
def stop_all(c):
    camera.unobserve_all()
    scanner.stop()
    robot.stop()
    print("Shutdown complete.")

stop_button.on_click(stop_all)
camera.observe(execute, names='value')
