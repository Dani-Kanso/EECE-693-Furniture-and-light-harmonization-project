"""
detector.py

This module uses a pretrained YOLOv8 model to detect beds in room images.

YOLOv8 is a real-time object detector trained on the COCO dataset, which
includes 80 object categories. "Bed" is COCO class ID 59. We use the
nano variant (yolov8n.pt) because it is fast and small — good enough for
this pipeline since we only need rough bounding boxes.

Typical usage:
    from detector import load_model, detect_bed, create_mask_from_bbox

    model = load_model()                   # load once, reuse many times
    result = detect_bed("room.jpg", model)
    if result:
        mask = create_mask_from_bbox((H, W), result["bbox"])
"""

import sys
import numpy as np
import cv2
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# COCO class ID for "bed". YOLO was trained on COCO, so each detected object
# has an integer class label. 59 corresponds to bed.
# ---------------------------------------------------------------------------
BED_CLASS_ID = 59

# Default model weights file. ultralytics will auto-download this on first
# use if it is not already cached locally.
DEFAULT_MODEL_WEIGHTS = "yolov8n.pt"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights: str = DEFAULT_MODEL_WEIGHTS) -> YOLO:
    """
    Load a YOLOv8 model from the given weights file and return it.

    We expose this as a separate function so callers can load the model once
    and pass it to detect_bed() repeatedly, avoiding the overhead of
    reloading the model for every image.

    Parameters
    ----------
    weights : str
        Path to (or name of) the YOLOv8 weights file.
        If you pass a bare filename like "yolov8n.pt", ultralytics will look
        for it in the local directory or its cache, and download it
        automatically if missing.

    Returns
    -------
    YOLO
        The loaded YOLOv8 model object, ready for inference.
    """
    # YOLO() is the main class from the ultralytics package.
    # It handles loading weights, setting up the neural network, etc.
    model = YOLO(weights)
    return model


# ---------------------------------------------------------------------------
# Bed detection
# ---------------------------------------------------------------------------

def detect_bed(
    image_path: str,
    model: YOLO = None,
    confidence_threshold: float = 0.3,
) -> dict | None:
    """
    Detect a bed in a single room image using YOLOv8.

    How it works:
      1. Load the image from disk.
      2. Run YOLOv8 inference to get bounding boxes for all detected objects.
      3. Filter to keep only detections whose class is "bed" (ID 59) and
         whose confidence score meets the threshold.
      4. If multiple beds are found, keep the one with the highest confidence
         (this handles rare cases where the model fires twice on the same bed).
      5. Return a dict with the bounding box, confidence, and area, or None
         if no bed is found.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the room image file.
    model : YOLO, optional
        A pre-loaded YOLO model. If None, a new model is loaded from
        DEFAULT_MODEL_WEIGHTS. Passing in a pre-loaded model is much faster
        when processing many images.
    confidence_threshold : float
        Minimum confidence score (0–1) required to accept a detection.
        Lower values catch more beds but may include false positives.

    Returns
    -------
    dict or None
        If a bed is found:
            {
                "bbox":       (x1, y1, x2, y2),  # integers, pixel coords
                "confidence": float,               # 0.0 – 1.0
                "area":       int,                 # (x2-x1) * (y2-y1) pixels
            }
        If no bed is found, returns None.
    """
    # --- Load model if the caller did not provide one ----------------------
    # Loading a model takes time, so callers who process many images should
    # call load_model() once and pass it here rather than relying on this
    # fallback path.
    if model is None:
        model = load_model()

    # --- Load the image with OpenCV ----------------------------------------
    # cv2.imread returns a NumPy array in BGR format (not RGB).
    # YOLOv8 / ultralytics can accept both file paths and NumPy arrays,
    # but we read it ourselves so we can validate that it loaded correctly.
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # --- Run YOLOv8 inference -----------------------------------------------
    # model() returns a list of Results objects, one per input image.
    # verbose=False suppresses the per-image console log lines.
    results = model(image, verbose=False)

    # results[0] corresponds to our single input image.
    # .boxes contains all detected bounding boxes for that image.
    boxes = results[0].boxes

    # --- Filter for bed detections -----------------------------------------
    bed_detections = []

    for box in boxes:
        # box.cls is a tensor containing the class ID of this detection.
        # .item() converts a single-element tensor to a plain Python number.
        class_id = int(box.cls.item())

        # box.conf is the confidence score (how sure the model is).
        confidence = float(box.conf.item())

        # Skip this detection if it is not a bed, or if confidence is too low.
        if class_id != BED_CLASS_ID:
            continue
        if confidence < confidence_threshold:
            continue

        # box.xyxy gives the bounding box as [x1, y1, x2, y2] in a tensor.
        # We convert to a plain Python list of floats first.
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Cast to integers because pixel coordinates must be whole numbers.
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Compute the area of this bounding box in pixels.
        area = (x2 - x1) * (y2 - y1)

        bed_detections.append({
            "bbox":       (x1, y1, x2, y2),
            "confidence": confidence,
            "area":       area,
        })

    # --- If nothing found, return None -------------------------------------
    if not bed_detections:
        return None

    # --- If multiple beds found, keep the most confident one ---------------
    # We sort by confidence in descending order and take the first element.
    bed_detections.sort(key=lambda d: d["confidence"], reverse=True)
    best = bed_detections[0]

    return best


# ---------------------------------------------------------------------------
# Mask creation
# ---------------------------------------------------------------------------

def create_mask_from_bbox(
    image_size: tuple,
    bbox: tuple,
    padding: int = 10,
) -> np.ndarray:
    """
    Create a binary mask image from a bounding box.

    The mask is a 2-D NumPy array with the same height and width as the
    original image. Pixels inside the (padded) bounding box are set to 255
    (white); everything else is 0 (black).

    Why do we need a mask?
      The next stage in the pipeline is LaMa (a large-mask inpainting model).
      LaMa expects a mask that tells it *which pixels to erase and
      regenerate*. We draw a white rectangle over the bed area so LaMa
      knows to fill that region with something else (e.g., new furniture).

    Why add padding?
      The bounding box from YOLO is tight around the detected object. A small
      amount of extra padding ensures LaMa also receives the immediate
      surroundings of the bed, giving it better context for inpainting and
      producing cleaner blending at the edges.

    Parameters
    ----------
    image_size : tuple
        (height, width) of the original image in pixels.
    bbox : tuple
        (x1, y1, x2, y2) bounding box coordinates as integers.
    padding : int
        Number of pixels to expand the bounding box in every direction.
        The expanded region is clamped so it never goes outside the image.

    Returns
    -------
    np.ndarray
        uint8 array of shape (height, width) with values 0 or 255.
    """
    height, width = image_size
    x1, y1, x2, y2 = bbox

    # --- Apply padding, clamped to image boundaries ------------------------
    # max(0, ...) prevents negative coordinates (top/left edge).
    # min(dim-1, ...) prevents coordinates beyond the image (bottom/right).
    x1_padded = max(0, x1 - padding)
    y1_padded = max(0, y1 - padding)
    x2_padded = min(width - 1, x2 + padding)
    y2_padded = min(height - 1, y2 + padding)

    # --- Create a blank (all-zero, black) mask of the same size as the image
    # np.zeros returns a float array by default; dtype=np.uint8 gives us
    # values in [0, 255], which is what image formats expect.
    mask = np.zeros((height, width), dtype=np.uint8)

    # --- Fill the bounding box region with white (255) ---------------------
    # NumPy array indexing is [row_start:row_end, col_start:col_end].
    # Rows correspond to the y-axis; columns correspond to the x-axis.
    mask[y1_padded:y2_padded + 1, x1_padded:x2_padded + 1] = 255

    return mask


# ---------------------------------------------------------------------------
# Quick smoke-test when run as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Simple command-line test. Usage:
        python detector.py /path/to/room_image.jpg

    This will:
      1. Load the YOLOv8 model.
      2. Run bed detection on the supplied image.
      3. Print the result dict (or a "no bed found" message).
      4. If a bed is found, create a mask and save it next to the input
         image as  <original_name>_mask.png  for visual inspection.
    """
    if len(sys.argv) < 2:
        print("Usage: python detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Loading YOLOv8 model ...")
    model = load_model()

    print(f"Running detection on: {image_path}")
    result = detect_bed(image_path, model=model)

    if result is None:
        print("No bed detected in this image.")
    else:
        print("Bed detected!")
        print(f"  Bounding box : {result['bbox']}")
        print(f"  Confidence   : {result['confidence']:.3f}")
        print(f"  Area (px)    : {result['area']}")

        # Load image to get its dimensions, then create and save the mask.
        image = cv2.imread(image_path)
        h, w = image.shape[:2]  # shape is (height, width, channels)
        mask = create_mask_from_bbox((h, w), result["bbox"])

        # Build an output path like "/some/dir/room_mask.png"
        # by replacing the extension of the original filename.
        import os
        base, _ = os.path.splitext(image_path)
        mask_path = base + "_mask.png"
        cv2.imwrite(mask_path, mask)
        print(f"  Mask saved to: {mask_path}")
