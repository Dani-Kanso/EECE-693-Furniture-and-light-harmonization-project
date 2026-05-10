"""
preprocessing package for the furniture placement GAN project.

This package handles all preprocessing stages of the pipeline:
  - scanning the dataset to find bedroom rooms (scanner)
  - detecting beds in room images and generating bounding boxes (detector)
  - inpainting (removing) detected furniture to create clean room backgrounds (inpainter)
"""

# --- Scanner: finds bedroom rooms in the 3D-FRONT dataset ---
from .scanner import scan_bedrooms

# --- Detector: loads the YOLO model, detects beds, and creates inpainting masks ---
from .detector import load_model, detect_bed, create_mask_from_bbox

# --- Inpainter: loads the LaMa model, runs inpainting, and saves results ---
from .inpainter import load_inpainter, inpaint_furniture, save_result
