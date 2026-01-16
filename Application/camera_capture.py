# ─── Cell 1 - Imports ───────────────────────────────────────────────
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn.functional as F
import PIL.Image
import numpy as np
import traitlets
from IPython.display import display
import ipywidgets.widgets as widgets
from jetbot import Camera, bgr8_to_jpeg

# ─── Cell 2 - Load your trained model ───────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  # 4 classes

# Load your trained weights
model.load_state_dict(torch.load('resnet18_apple_disease.pth'))
model = model.to(device)
model.eval()

print("Model loaded successfully!")

# ─── Cell 3 - Preprocessing (must match training) ───────────────────
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── Cell 4 - Your REAL class names (VERY IMPORTANT!) ───────────────
# Replace these with your actual 4 apple disease classes
class_names = [
    'Healthy', 
    'Apple Scab', 
    'Black Rot', 
    'Cedar Apple Rust'
]  # ← CHANGE THESE TO YOUR REAL CLASSES!

# ─── Cell 5 - Widgets & Camera Control ──────────────────────────────
camera = None  # will be initialized when started
image_widget = widgets.Image(format='jpeg', width=320, height=320)
label_widget = widgets.Label(value='Camera stopped - press Start')

btn_start = widgets.Button(description='Start Camera', button_style='success')
btn_stop  = widgets.Button(description='Stop Camera', button_style='danger')
btn_stop.disabled = True

# Status label
status_label = widgets.Label(value='Ready')

# ─── Prediction function ────────────────────────────────────────────
def live_predict(change):
    if camera is None or not camera.running:
        return
        
    image = change['new']                     # BGR numpy (224,224,3)
    image_rgb = image[..., ::-1]              # BGR → RGB
    
    image_pil = PIL.Image.fromarray(image_rgb)
    input_tensor = preprocess(image_pil)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_batch)
        probabilities = F.softmax(output, dim=1)[0]
        confidence, class_id = torch.max(probabilities, 0)
    
    pred_class = class_names[class_id.item()]
    conf = confidence.item()
    
    label_widget.value = f'→ {pred_class}  ({conf:.3f})'

# ─── Camera control functions ───────────────────────────────────────
def start_camera(b):
    global camera
    if camera is not None:
        camera.stop()
    
    try:
        camera = Camera.instance(width=224, height=224)
        camera.running = True
        
        # Connect prediction
        camera.observe(live_predict, names='value')
        
        # Connect video stream
        traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)
        
        btn_start.disabled = True
        btn_stop.disabled = False
        status_label.value = 'Camera running'
        
    except Exception as e:
        status_label.value = f'Error starting camera: {str(e)}'

def stop_camera(b):
    global camera
    if camera is not None:
        camera.unobserve(live_predict, names='value')
        camera.stop()
        camera = None
    
    btn_start.disabled = False
    btn_stop.disabled = True
    status_label.value = 'Camera stopped'
    label_widget.value = 'Camera stopped - press Start'

# ─── Connect buttons ────────────────────────────────────────────────
btn_start.on_click(start_camera)
btn_stop.on_click(stop_camera)

# ─── Layout ─────────────────────────────────────────────────────────
controls = widgets.HBox([btn_start, btn_stop, status_label])
display(widgets.VBox([
    controls,
    widgets.HBox([image_widget, label_widget]),
    widgets.Label('Apple Disease Detection - Live Camera')
]))
