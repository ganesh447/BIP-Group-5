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
model_path = 'final_model.pth' 
speed_gain = 0.15
steering_gain = 0.12
steering_kd = 0.04

# LiDAR Parameters
RECOVERY_DIST = 160.0 
STOP_DIST = 220.0     
AVOID_DIST = 500.0  
SAFE_CORRIDOR = 220.0 

# --- 2. LIDAR THREAD (Adjusted for 60° Cone) ---
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
                            # --- 60 DEGREE FILTER (330 to 30) ---
                            if (angle > 330 or angle < 30):
                                if dist < self.min_dist: self.min_dist = dist
                                rad = np.deg2rad(angle)
                                x_coord = dist * np.sin(rad)
                                danger_x.append(x_coord)
                    
                    self.avg_x_raw = np.mean(danger_x) if len(danger_x) > 5 else 0.0
            except: self.lidar.clean_input()

    def stop(self):
        self.stop_signal = True
        self.lidar.stop_motor()
        self.lidar.disconnect()

# --- 3. SETUP ---
scanner = LidarScanner()
scanner.start()
robot = Robot()
camera = Camera.instance(width=224, height=224)

# Model Load
device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load(model_path))
model = model.to(device).eval().half()
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

# UI WIDGETS
image_widget = widgets.Image(format='jpeg', width=224, height=224)
label_widget = widgets.Label(value="Status: Ready")
stop_button = widgets.Button(description='STOP ROBOT', button_style='danger', layout={'width': '100%'})
display(widgets.VBox([image_widget, label_widget, stop_button]))

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float().permute(2, 0, 1).to(device).half() / 255.0
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# --- 4. EXECUTION LOOP ---
last_x = 0.0
is_recovering = False

def execute(change):
    global last_x, is_recovering
    image = change['new']
    dist = scanner.min_dist
    obs_x = scanner.avg_x_raw

    # A. RECOVERY LOGIC (Backwards)
    if dist < RECOVERY_DIST or is_recovering:
        is_recovering = True
        robot.backward(0.1)
        label_widget.value = "STATE: RECOVERING (REVERSE)"
        if dist > (STOP_DIST + 150): 
            is_recovering = False
            robot.stop()
        return

    # B. AI LANE PREDICTION
    with torch.no_grad():
        output = model(preprocess(image)).flatten()
    ai_x = float(output[0])

    # C. GRADUAL SMOOTH NUDGE
    path_mm = ai_x * 200.0 
    delta_x = obs_x - path_mm
    
    gradual_nudge = 0.0
    speed_scaler = 1.0 
    
    if dist < AVOID_DIST and abs(delta_x) < SAFE_CORRIDOR:
        proximity_weight = np.clip((AVOID_DIST - dist) / (AVOID_DIST - STOP_DIST), 0, 1)
        overlap_weight = np.clip((SAFE_CORRIDOR - abs(delta_x)) / SAFE_CORRIDOR, 0, 1)
        
        combined_influence = proximity_weight * overlap_weight
        direction = -1.0 if delta_x > 0 else 1.0
        
        # Proportional gradual nudge
        gradual_nudge = direction * combined_influence * 0.20
        # Smoothly slow down based on how blocked the path is
        speed_scaler = 1.0 - (combined_influence * 0.6)

    # D. COMBINED CONTROL
    change_in_x = ai_x - last_x
    ai_steering = (ai_x * steering_gain) + (change_in_x * steering_kd)
    total_steering = ai_steering + gradual_nudge
    last_x = ai_x
    
    current_speed = speed_gain * speed_scaler
    
    l_v = max(min(current_speed + total_steering, 1.0), 0.0)
    r_v = max(min(current_speed - total_steering, 1.0), 0.0)
    robot.set_motors(l_v, r_v)

    # E. VISUALS
    px = int(112 + ai_x * 112)
    cv2.line(image, (112, 224), (px, 112), (0, 255, 0), 2)
    if abs(gradual_nudge) > 0.02:
        target_px = int(112 + (ai_x + gradual_nudge) * 112)
        cv2.arrowedLine(image, (112, 180), (target_px, 130), (0, 0, 255), 2)
    
    label_widget.value = f"Speed: {int(speed_scaler*100)}% | Nudge: {gradual_nudge:.2f}"
    image_widget.value = bgr8_to_jpeg(image)

# --- 5. CLEANUP ---
def stop_all(c):
    camera.unobserve_all()
    scanner.stop()
    robot.stop()
    print("Emergency Stop triggered.")

stop_button.on_click(stop_all)
camera.observe(execute, names='value')
