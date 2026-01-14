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

# Pure Pursuit Parameters
V_MAX = 0.35        # Max speed on straights (increased from 0.15)
V_MIN = 0.18        # Min speed for sharp curves
GAIN_STEER = 0.75   # How aggressively to follow the arc
LOOKAHEAD_BIAS = 1.2 # Multiplier to look further ahead on the Y-axis

# LiDAR Parameters (Kept from your original logic)
RECOVERY_DIST = 160.0 
STOP_DIST = 220.0     
AVOID_DIST = 500.0  
SAFE_CORRIDOR = 220.0 

# --- 2. LIDAR THREAD (Unchanged) ---
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

# Model Load (Optimized for XY Output)
device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2) # Ensure model trained for X and Y
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
is_recovering = False

def execute(change):
    global is_recovering
    image = change['new']
    dist = scanner.min_dist
    obs_x = scanner.avg_x_raw

    # A. RECOVERY LOGIC (Unchanged)
    if dist < RECOVERY_DIST or is_recovering:
        is_recovering = True
        robot.backward(0.12)
        label_widget.value = "STATE: RECOVERING (REVERSE)"
        if dist > (STOP_DIST + 150): 
            is_recovering = False
            robot.stop()
        return

    # B. AI LANE PREDICTION (X, Y)
    with torch.no_grad():
        output = model(preprocess(image)).flatten()
    ai_x = float(output[0])
    ai_y = float(output[1])

    # C. PURE PURSUIT GEOMETRY
    # L is the distance to target; ty is biased to look further ahead for stability
    L = np.sqrt(ai_x**2 + (ai_y * LOOKAHEAD_BIAS)**2)
    
    # Curvature: k = 2x / L^2
    curvature = (2.0 * ai_x) / (L**2 + 1e-6)
    
    # D. ADAPTIVE CONTROL
    total_steering = curvature * GAIN_STEER
    
    # Speed scaling: Slow down only for high curvature (sharp turns)
    speed_scaler = V_MAX - (abs(curvature) * (V_MAX - V_MIN))
    speed_scaler = max(V_MIN, speed_scaler)

    # E. LIDAR SLOWDOWN (Nudge influence)
    # If LiDAR sees obstacle in safe corridor, reduce speed further
    path_mm = ai_x * 200.0 
    delta_x = obs_x - path_mm
    if dist < AVOID_DIST and abs(delta_x) < SAFE_CORRIDOR:
        speed_scaler *= 0.5 # Half speed if path is blocked

    # F. MOTOR DRIVE
    l_v = max(min(speed_scaler + total_steering, 1.0), 0.0)
    r_v = max(min(speed_scaler - total_steering, 1.0), 0.0)
    robot.set_motors(l_v, r_v)

    # G. VISUALS
    # Green line shows the Pure Pursuit Target
    px = int(112 + ai_x * 112)
    py = int(224 - (ai_y * 224))
    cv2.line(image, (112, 224), (px, py), (0, 255, 0), 2)
    cv2.circle(image, (px, py), 5, (0, 255, 0), -1)
    
    label_widget.value = f"Velocity: {speed_scaler:.2f} | Curvature: {curvature:.2f}"
    image_widget.value = bgr8_to_jpeg(image)

# --- 5. CLEANUP ---
def stop_all(c):
    camera.unobserve_all()
    scanner.stop()
    robot.stop()
    print("Emergency Stop triggered.")

stop_button.on_click(stop_all)
camera.observe(execute, names='value')
