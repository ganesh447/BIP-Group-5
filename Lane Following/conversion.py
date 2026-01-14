import xml.etree.ElementTree as ET
import os
import shutil
import uuid

# Config
XML_FILE = 'annotations.xml'
IMG_DIR = 'images'
OUT_DIR = 'dataset_xy'

if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

tree = ET.parse(XML_FILE)
root = tree.getroot()

print("Converting CVAT points to JetBot filenames...")

for img_tag in root.findall('image'):
    name = img_tag.get('name')
    point = img_tag.find('points')
    
    if point is not None:
        # Get coordinates from XML
        coords = point.get('points').split(',')
        x = int(float(coords[0]))
        y = int(float(coords[1]))
        
        # JetBot format: xy_X_Y_UUID.jpg
        # We pad with 3 zeros so the notebook reads it correctly
        new_name = f"xy_{x:03d}_{y:03d}_{uuid.uuid4().hex}.jpg"
        
        shutil.copy(os.path.join(IMG_DIR, name), os.path.join(OUT_DIR, new_name))

print(f"Finished! Dataset created in '{OUT_DIR}'")