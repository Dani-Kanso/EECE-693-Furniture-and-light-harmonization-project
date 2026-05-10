"""
pipeline.py — Main preprocessing pipeline for the furniture placement GAN project.

This script ties together the scanner, detector, and inpainter modules to produce
training triplets:
  - input:     inpainted room (bed removed)
  - furniture: cropped bed image from the render (isolated on white background)
  - target:    original room with bed (ground truth)

These triplets will later train a GAN to learn how to place a specific piece of
furniture into a room, given the room image and the furniture image.

Usage (from project root):
    python -m src.preprocessing.pipeline
    python src/preprocessing/pipeline.py
"""

import sys
import os

# Add project root to path so imports work both ways:
# - When run as `python src/preprocessing/pipeline.py` (direct script)
# - When run as `python -m src.preprocessing.pipeline` (module)
# Without this, Python can't find the `src` package when running directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.scanner import scan_bedrooms
from src.preprocessing.detector import load_model, detect_bed, create_mask_from_bbox
from src.preprocessing.inpainter import load_inpainter, inpaint_furniture, save_result

import argparse
import random
import csv
from pathlib import Path

import pandas as pd
import cv2
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """
    Define and parse all command-line arguments for the pipeline.

    This lets you run the pipeline with different settings without editing code.
    For example: python -m src.preprocessing.pipeline --confidence 0.5 --seed 7
    """
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline: detect beds, inpaint them out, and save training pairs."
    )

    # Where is the raw dataset?
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/raw/3D-FRONT-TEST-RENDER",
        help="Path to the raw 3D-FRONT-TEST-RENDER dataset directory."
    )

    # Where should processed output go?
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Root directory for processed train/val/test output."
    )

    # YOLOv8 detection confidence threshold.
    # Lower = more detections (but more false positives).
    # Higher = fewer detections (but more misses).
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="YOLOv8 confidence threshold for bed detection (0.0 to 1.0)."
    )

    # Padding added around the detected bounding box before creating the mask.
    # This ensures the mask covers the full furniture piece, not just the tight box.
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Pixel padding added around the bounding box when creating the inpainting mask."
    )

    # How to split rooms into train / val / test.
    # These three must sum to 1.0.
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of rooms to use for training (e.g., 0.8 = 80%%)."
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of rooms to use for validation (e.g., 0.1 = 10%%)."
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of rooms to use for testing (e.g., 0.1 = 10%%)."
    )

    # Random seed for reproducibility.
    # Using the same seed guarantees the same train/val/test split every run.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val/test splitting."
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Train / val / test splitting
# ---------------------------------------------------------------------------

def split_rooms(rooms, train_ratio, val_ratio, test_ratio, seed):
    """
    Randomly shuffle rooms and split them into train, val, and test sets.

    IMPORTANT: We split by ROOM, not by individual images. This means all 11
    camera angles for a given room end up in the same split. If we split by
    image instead, the model could "memorize" a room from one angle during
    training and be evaluated on a different angle of the same room — that
    would be data leakage, making our evaluation results unrealistically good.

    Args:
        rooms         : list of room dicts from scan_bedrooms()
        train_ratio   : fraction for training (e.g., 0.8)
        val_ratio     : fraction for validation (e.g., 0.1)
        test_ratio    : fraction for testing (e.g., 0.1)
        seed          : integer seed for reproducible shuffling

    Returns:
        dict with keys "train", "val", "test", each mapping to a list of rooms
    """
    # Validate that ratios sum to ~1.0 (allow tiny floating point error)
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total:.4f}"
        )

    # Make a copy so we don't mutate the original list
    shuffled = rooms[:]
    random.seed(seed)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    # test gets whatever is left over (avoids off-by-one from float rounding)
    n_test  = n - n_train - n_val

    train_rooms = shuffled[:n_train]
    val_rooms   = shuffled[n_train : n_train + n_val]
    test_rooms  = shuffled[n_train + n_val :]

    print(f"Split: {len(train_rooms)} train | {len(val_rooms)} val | {len(test_rooms)} test rooms")

    return {
        "train": train_rooms,
        "val":   val_rooms,
        "test":  test_rooms,
    }


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------

def create_output_dirs(output_dir):
    """
    Create the directory structure for processed output.

    Expected structure:
        data/processed/
            train/input/      — inpainted rooms (bed removed)
            train/target/     — original rerenders (bed present)
            train/furniture/  — cropped bed from render (isolated furniture)
            val/input/
            val/target/
            val/furniture/
            test/input/
            test/target/
            test/furniture/

    Using exist_ok=True means this won't fail if the directories already exist.
    """
    output_path = Path(output_dir)
    for split in ["train", "val", "test"]:
        (output_path / split / "input").mkdir(parents=True, exist_ok=True)
        (output_path / split / "target").mkdir(parents=True, exist_ok=True)
        (output_path / split / "furniture").mkdir(parents=True, exist_ok=True)
    print(f"Output directories ready under: {output_path.resolve()}")


# ---------------------------------------------------------------------------
# Core processing loop
# ---------------------------------------------------------------------------

def process_split(split_name, rooms, output_dir, model, inpainter, confidence, padding):
    """
    Process all rooms (and all angles per room) for a given data split.

    For each room × angle pair:
      1. Build the path to the rerender image
      2. Detect the bed using YOLOv8
      3. If no bed found, skip
      4. Create an inpainting mask around the bounding box
      5. Run LaMa inpainting to erase the bed
      6. Crop the bed from the render image (furniture on white background)
      7. Save the inpainted image as the INPUT (condition)
      8. Save the original rerender as the TARGET (ground truth)
      9. Save the cropped furniture image as the FURNITURE reference
      10. Record metadata

    Args:
        split_name  : "train", "val", or "test"
        rooms       : list of room dicts for this split
        output_dir  : root output directory (string or Path)
        model       : loaded YOLOv8 model (from load_model())
        inpainter   : loaded LaMa inpainter (from load_inpainter())
        confidence  : detection confidence threshold
        padding     : bounding box mask padding in pixels

    Returns:
        metadata_rows : list of dicts (one per successfully processed pair)
        skipped_count : number of (room, angle) pairs that had no bed detected
        error_count   : number of (room, angle) pairs that failed due to an error
    """
    output_path = Path(output_dir)
    metadata_rows = []
    skipped_count = 0
    error_count   = 0

    total_rooms = len(rooms)

    for room_idx, room in enumerate(rooms, start=1):
        uuid      = room["uuid"]
        room_name = room["room_name"]
        room_path = room["room_path"]
        angles    = room["angles"]
        total_angles = len(angles)

        for angle_idx, angle in enumerate(angles, start=1):
            # Progress indicator — e.g., "  [train] Room 12/200 — angle 3/11"
            print(f"  [{split_name}] Room {room_idx}/{total_rooms} — angle {angle_idx}/{total_angles} "
                  f"({uuid}_{room_name}, angle {angle})")

            # Build the path to the rerender image for this angle.
            # The rerender is the camera-rendered image used as ground truth.
            rerender_path = os.path.join(room_path, f"rerender_{angle}.webp")

            # Wrap each image in a try/except so one bad file doesn't crash everything.
            try:
                # --- Step 1: Check the file exists ---
                if not os.path.isfile(rerender_path):
                    print(f"    WARNING: File not found, skipping — {rerender_path}")
                    skipped_count += 1
                    continue

                # --- Step 2: Detect bed with YOLOv8 ---
                detection = detect_bed(rerender_path, model, confidence_threshold=confidence)

                # If no bed was detected, skip this image.
                # A missing bed could mean the angle doesn't show it, or it's a non-bedroom.
                if detection is None:
                    print(f"    No bed detected, skipping.")
                    skipped_count += 1
                    continue

                bbox       = detection["bbox"]        # [x1, y1, x2, y2]
                conf_score = detection["confidence"]

                # --- Step 3: Load image to get its size for mask creation ---
                # cv2.imread returns a numpy array; .shape gives (height, width, channels)
                img_cv = cv2.imread(rerender_path)
                if img_cv is None:
                    print(f"    ERROR: cv2 could not read image — {rerender_path}")
                    error_count += 1
                    continue

                image_size = (img_cv.shape[1], img_cv.shape[0])  # (width, height)

                # --- Step 4: Create binary mask from bounding box ---
                # The mask is a numpy array (uint8) where 255 = region to inpaint (bed),
                # and 0 = region to keep unchanged.
                mask = create_mask_from_bbox(image_size, bbox, padding=padding)

                # --- Step 5: Inpaint the bed region with LaMa ---
                # LaMa fills in the masked region using surrounding context (walls, floor, etc.)
                inpainted_image = inpaint_furniture(rerender_path, mask, inpainter)

                # --- Step 6: Crop the furniture from the rerender image ---
                # We crop the bed region directly from the rerender (photorealistic image)
                # using the same YOLOv8 bounding box. This gives us the actual bed with
                # correct colors, textures, lighting, and shadows — exactly how it appears
                # in the room. This cropped image is what the model will use as a reference
                # for what furniture to place.
                rerender_pil = PILImage.open(rerender_path).convert("RGB")
                # Crop using the bounding box (x1, y1, x2, y2) — PIL uses (left, upper, right, lower)
                furniture_image = rerender_pil.crop(bbox)

                # --- Step 7: Build output file paths ---
                # Use a naming scheme: {uuid}_{room_name}_{angle}.png
                # This uniquely identifies every image and makes it easy to match triplets.
                filename = f"{uuid}_{room_name}_{angle}.png"

                input_path     = output_path / split_name / "input"     / filename  # inpainted (no bed)
                target_path    = output_path / split_name / "target"    / filename  # original (with bed)
                furniture_path = output_path / split_name / "furniture" / filename  # cropped bed image

                # --- Step 8: Save the inpainted image (input for the GAN) ---
                save_result(inpainted_image, str(input_path))

                # --- Step 9: Save the original rerender (target / ground truth) ---
                # rerender_pil was already loaded above for furniture cropping, reuse it.
                rerender_pil.save(str(target_path))

                # --- Step 10: Save the cropped furniture image ---
                furniture_image.save(str(furniture_path))

                # --- Step 11: Record metadata for this triplet ---
                metadata_rows.append({
                    "uuid":           uuid,
                    "room_name":      room_name,
                    "angle":          angle,
                    "split":          split_name,
                    "input_path":     str(input_path),
                    "target_path":    str(target_path),
                    "furniture_path": str(furniture_path),
                    "bbox_x1":        bbox[0],
                    "bbox_y1":        bbox[1],
                    "bbox_x2":        bbox[2],
                    "bbox_y2":        bbox[3],
                    "confidence":     round(conf_score, 4),
                })

            except Exception as e:
                # Catch any unexpected error (corrupt file, OOM, etc.) and keep going.
                print(f"    ERROR processing {rerender_path}: {e}")
                error_count += 1
                continue

    return metadata_rows, skipped_count, error_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main function that orchestrates the entire preprocessing pipeline.

    Flow:
        1. Parse args
        2. Scan dataset for bedroom rooms
        3. Split rooms into train / val / test
        4. Create output directories
        5. Load models once (expensive — do this before the loops)
        6. Process each split
        7. Save metadata CSV
        8. Print summary
    """

    # ---- Parse command-line arguments ----------------------------------------
    args = parse_args()

    # Validate split ratios before doing any heavy work
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"ERROR: train_ratio + val_ratio + test_ratio = {total_ratio:.4f}, must equal 1.0")
        sys.exit(1)

    print("=" * 60)
    print("Furniture Placement GAN — Preprocessing Pipeline")
    print("=" * 60)
    print(f"Dataset root : {args.dataset_root}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Confidence   : {args.confidence}")
    print(f"Mask padding : {args.padding}px")
    print(f"Split ratios : train={args.train_ratio} / val={args.val_ratio} / test={args.test_ratio}")
    print(f"Random seed  : {args.seed}")
    print()

    # ---- Step 1: Scan the dataset for bedroom rooms --------------------------
    print("Scanning dataset for bedroom rooms...")
    rooms = scan_bedrooms(args.dataset_root)
    print(f"Found {len(rooms)} bedroom rooms.\n")

    if len(rooms) == 0:
        print("ERROR: No rooms found. Check --dataset_root path.")
        sys.exit(1)

    # ---- Step 2: Split rooms into train / val / test -------------------------
    # We do this BEFORE loading images so the split is pure — based only on room IDs.
    print("Splitting rooms into train / val / test sets...")
    splits = split_rooms(
        rooms,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print()

    # ---- Step 3: Create output directories -----------------------------------
    create_output_dirs(args.output_dir)
    print()

    # ---- Step 4: Load models once --------------------------------------------
    # Loading models is slow (downloading weights, allocating GPU memory, etc.).
    # We load them here once and reuse across all images — much faster than
    # loading per-image.

    print("Loading YOLOv8 bed detection model...")
    model = load_model()
    print("YOLOv8 model loaded.\n")

    print("Loading LaMa inpainting model...")
    inpainter = load_inpainter()
    print("LaMa inpainter loaded.\n")

    # ---- Step 5: Process each split ------------------------------------------
    all_metadata = []       # will collect rows from all three splits
    summary = {}            # final stats per split

    for split_name in ["train", "val", "test"]:
        split_rooms_list = splits[split_name]
        print(f"\n--- Processing split: {split_name.upper()} ({len(split_rooms_list)} rooms) ---")

        metadata_rows, skipped, errors = process_split(
            split_name=split_name,
            rooms=split_rooms_list,
            output_dir=args.output_dir,
            model=model,
            inpainter=inpainter,
            confidence=args.confidence,
            padding=args.padding,
        )

        all_metadata.extend(metadata_rows)
        summary[split_name] = {
            "pairs":   len(metadata_rows),
            "skipped": skipped,
            "errors":  errors,
        }

        print(f"  Done. Pairs saved: {len(metadata_rows)} | Skipped (no bed): {skipped} | Errors: {errors}")

    # ---- Step 6: Save metadata CSV -------------------------------------------
    # The CSV lets us inspect what was processed, cross-reference splits,
    # and debug by looking up specific rooms or angles.
    metadata_csv_path = os.path.join(args.output_dir, "metadata.csv")

    if all_metadata:
        df = pd.DataFrame(all_metadata, columns=[
            "uuid", "room_name", "angle", "split",
            "input_path", "target_path", "furniture_path",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "confidence",
        ])
        df.to_csv(metadata_csv_path, index=False)
        print(f"\nMetadata saved to: {metadata_csv_path}  ({len(df)} rows)")
    else:
        print("\nWARNING: No metadata to save — no pairs were successfully processed.")

    # ---- Step 7: Print final summary -----------------------------------------
    print("\n" + "=" * 60)
    print("Pipeline complete — Summary")
    print("=" * 60)

    total_pairs   = 0
    total_skipped = 0
    total_errors  = 0

    for split_name in ["train", "val", "test"]:
        s = summary[split_name]
        print(f"  {split_name:6s} : {s['pairs']:5d} pairs | {s['skipped']:4d} skipped | {s['errors']:4d} errors")
        total_pairs   += s["pairs"]
        total_skipped += s["skipped"]
        total_errors  += s["errors"]

    print(f"  {'TOTAL':6s} : {total_pairs:5d} pairs | {total_skipped:4d} skipped | {total_errors:4d} errors")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Script entry point guard
# ---------------------------------------------------------------------------

# This block runs only when the script is called directly (not when imported).
# It's the standard Python pattern for scripts that are also importable as modules.
if __name__ == "__main__":
    main()
