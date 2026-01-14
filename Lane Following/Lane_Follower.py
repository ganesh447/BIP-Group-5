import torch
import torchvision
import cv2
import numpy as np
import time
from jetbot import Robot, Camera, bgr8_to_jpeg
import ipywidgets.widgets as widgets
from IPython.display import display

# --- 1. CONFIGURATION ---
model_path = 'best_steering_models.pth'
speed_gain = 0.15
steering_gain = 0.04

# --- 2. MODEL SETUP ---
device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load(model_path))
model = model.to(device).eval().half() # FP16 for speed

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

# --- 5. THE EXECUTION FUNCTION ---
def execute(change):
    image = change['new']
    
    # Run Inference
    with torch.no_grad():
        output = model(preprocess(image)).flatten()
    
    x = float(output[0])
    
    # Motor Control Logic
    steering = x * steering_gain
    left_v = max(min(speed_gain + steering, 1.0), 0.0)
    right_v = max(min(speed_gain - steering, 1.0), 0.0)
    robot.set_motors(left_v, right_v)
    
    # Visualization
    px = int(112 + x * 112)
    cv2.circle(image, (px, 112), 8, (0, 255, 0), -1)
    image_widget.value = bgr8_to_jpeg(image)

# --- 6. STOP BUTTON LOGIC ---
def stop_all(c):
    camera.unobserve_all()
    time.sleep(0.5)
    robot.stop()
    print("Emergency Stop: Robot Halted and Camera Released.")

stop_button.on_click(stop_all)

# --- 7. START ---
camera.observe(execute, names='value')
print("Robot is LIVE. Click the RED BUTTON to stop.")
