"""
inpainter.py

This module wraps the LaMa (Large Mask) inpainting model via the
`simple-lama-inpainting` package. Its job is to remove furniture from
room images by filling in the masked regions with realistic background.

Typical usage:
    1. Call load_inpainter() once at startup to get a reusable model object.
    2. Call inpaint_furniture() for each image you want to process.
    3. Call save_result() to write the output to disk.
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from simple_lama_inpainting import SimpleLama


# ---------------------------------------------------------------------------
# 1. Model loader
# ---------------------------------------------------------------------------

def load_inpainter() -> SimpleLama:
    """
    Initialize and return the LaMa inpainting model.

    Loading the model is expensive (it downloads / reads weights into memory),
    so this function exists so you can do it once and pass the same object to
    every call of inpaint_furniture() instead of reloading on each image.

    Returns
    -------
    SimpleLama
        A ready-to-use LaMa inpainting model instance.
    """
    # SimpleLama() handles all weight loading internally.
    # No configuration arguments are needed for default usage.
    print("[inpainter] Loading LaMa model...")
    inpainter = SimpleLama()
    print("[inpainter] LaMa model loaded successfully.")
    return inpainter


# ---------------------------------------------------------------------------
# 2. Core inpainting function
# ---------------------------------------------------------------------------

def inpaint_furniture(
    image_path: str,
    mask: np.ndarray,
    inpainter: SimpleLama = None,
) -> Image.Image:
    """
    Remove furniture from a room image using LaMa inpainting.

    The mask tells the model *which pixels to fill in*. White pixels (255)
    mark the furniture areas that should be replaced with realistic background;
    black pixels (0) are left untouched.

    Parameters
    ----------
    image_path : str
        Path to the room RGB image (any format Pillow can open: PNG, JPEG, …).
    mask : np.ndarray
        A 2-D uint8 numpy array the same height/width as the image.
        Pixel values must be either 0 (keep) or 255 (inpaint).
    inpainter : SimpleLama, optional
        A pre-loaded LaMa model. If None, a new one is created on the fly
        (slower — prefer passing a model loaded with load_inpainter()).

    Returns
    -------
    PIL.Image.Image
        The inpainted image (room with furniture removed).
    """
    # --- Load the source image -----------------------------------------------
    # We open it as RGB to ensure a consistent 3-channel colour image,
    # regardless of whether the file on disk is RGBA or palettised.
    image = Image.open(image_path).convert("RGB")

    # --- Convert the numpy mask to a PIL grayscale image ---------------------
    # LaMa expects a PIL Image in mode "L" (8-bit greyscale).
    # np.uint8 ensures values stay in [0, 255] with no accidental float cast.
    mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L")

    # --- Safety check: mask and image must have the same spatial dimensions --
    if image.size != mask_pil.size:
        raise ValueError(
            f"Image size {image.size} does not match mask size {mask_pil.size}. "
            "They must be identical (width, height)."
        )

    # --- Lazily initialise the model if the caller did not supply one --------
    if inpainter is None:
        print("[inpainter] No inpainter supplied — creating a temporary one.")
        inpainter = load_inpainter()

    # --- Run LaMa inpainting -------------------------------------------------
    # simple_lama(image, mask) fills every white region in the mask with
    # content predicted by the neural network and returns a PIL Image.
    print(f"[inpainter] Inpainting '{image_path}' ...")
    result = inpainter(image, mask_pil)
    print("[inpainter] Inpainting complete.")

    return result


# ---------------------------------------------------------------------------
# 3. Result saver
# ---------------------------------------------------------------------------

def save_result(image: Image.Image, output_path: str) -> None:
    """
    Save an inpainted PIL Image to disk as a lossless PNG.

    PNG is chosen over JPEG because it is lossless — important when the
    inpainted images will be fed into downstream model training where
    compression artefacts could mislead the network.

    Parameters
    ----------
    image : PIL.Image.Image
        The inpainted image to save.
    output_path : str
        Destination file path. The extension should be '.png'; if it is
        something else the file will still be written as PNG data.
    """
    # Convert to a Path object for convenient directory manipulation.
    output_path = Path(output_path)

    # Create all parent directories if they do not already exist.
    # exist_ok=True means no error is raised if the directory is already there.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as PNG for lossless quality.
    image.save(str(output_path), format="PNG")
    print(f"[inpainter] Result saved to '{output_path}'.")


# ---------------------------------------------------------------------------
# 4. Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick CLI for running inpainting on a single image.

    Usage:
        python inpainter.py <image_path> <mask_path>

    The mask image should be an 8-bit greyscale PNG where white = furniture.
    The inpainted result is saved next to the original image with the suffix
    '_inpainted' added before the file extension.

    Example:
        python inpainter.py data/room.jpg data/room_mask.png
        # saves  data/room_inpainted.png
    """

    # --- Parse command-line arguments ----------------------------------------
    if len(sys.argv) != 3:
        print("Usage: python inpainter.py <image_path> <mask_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    mask_path  = sys.argv[2]

    # --- Validate that the input files actually exist ------------------------
    if not os.path.isfile(image_path):
        print(f"[error] Image not found: '{image_path}'")
        sys.exit(1)
    if not os.path.isfile(mask_path):
        print(f"[error] Mask not found: '{mask_path}'")
        sys.exit(1)

    # --- Load the mask from disk and convert to a numpy array ----------------
    # The mask file is opened in greyscale mode ("L") so we get a 2-D array.
    mask_image = Image.open(mask_path).convert("L")
    mask_array = np.array(mask_image, dtype=np.uint8)

    # --- Load the LaMa model once before processing --------------------------
    lama = load_inpainter()

    # --- Run inpainting -------------------------------------------------------
    inpainted = inpaint_furniture(image_path, mask_array, inpainter=lama)

    # --- Build the output path: same directory, stem + "_inpainted" + ".png" -
    src = Path(image_path)
    output_path = src.parent / f"{src.stem}_inpainted.png"

    # --- Save the result -------------------------------------------------------
    save_result(inpainted, str(output_path))
