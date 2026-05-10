"""
scanner.py

Scans the 3D-FRONT-TEST-RENDER dataset and filters for bedroom rooms.

The dataset has the following structure:
    <dataset_root>/
        <UUID>/                        # one folder per 3D scene
            <RoomType-ID>/             # one or more room folders per scene
                rerender_XXXX.webp     # rendered images at different camera angles
                render_XXXX.webp
                normal_XXXX.webp
                depth_XXXX.exr
                meta.json
            <AnotherRoomType-ID>/      # a UUID can have multiple room types

This module is the first step in the preprocessing pipeline: it discovers
which rooms are bedrooms and which camera angles are available for each.
"""

import os
import re


def scan_bedrooms(dataset_root: str) -> list[dict]:
    """
    Scan the 3D-FRONT-TEST-RENDER dataset and return metadata for every
    bedroom room found.

    Parameters
    ----------
    dataset_root : str
        Absolute path to the root of the 3D-FRONT-TEST-RENDER dataset,
        e.g. "/Users/karim/.../data/raw/3D-FRONT-TEST-RENDER"

    Returns
    -------
    list[dict]
        Each element describes one bedroom room and looks like:
        {
            "uuid":      "deaa4ac0-8d13-423f-930f-78fb25825e8d",
            "room_name": "Bedroom-5088",
            "room_path": "/full/path/to/Bedroom-5088/",
            "angles":    ["0000", "0001", ..., "0010"]
        }
    """

    # Collect results here — one entry per bedroom room
    bedrooms = []

    # ------------------------------------------------------------------ #
    # Step 1: Iterate over every entry inside the dataset root.           #
    # Each top-level entry is a UUID folder representing one 3D scene.   #
    # ------------------------------------------------------------------ #
    for uuid in sorted(os.listdir(dataset_root)):
        uuid_path = os.path.join(dataset_root, uuid)

        # Skip anything that is not a directory (e.g. stray files)
        if not os.path.isdir(uuid_path):
            continue

        # -------------------------------------------------------------- #
        # Step 2: Inside the UUID folder, look for room sub-folders.     #
        # A single scene can contain several rooms (e.g. a Bedroom AND   #
        # a CloakRoom), so we iterate over all of them.                  #
        # -------------------------------------------------------------- #
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)

            # Again, only process actual directories
            if not os.path.isdir(room_path):
                continue

            # -------------------------------------------------------------- #
            # Step 3: Filter — keep only rooms whose name contains "bedroom" #
            # (case-insensitive).  This captures:                             #
            #   "Bedroom-5088", "MasterBedroom-1095", "SecondBedroom-11739"  #
            # and excludes non-bedroom types like "CloakRoom", "LivingRoom". #
            # -------------------------------------------------------------- #
            if "bedroom" not in room_name.lower():
                continue

            # -------------------------------------------------------------- #
            # Step 4: Discover available camera angles for this room.        #
            # Angles are identified by the rerender_XXXX.webp files where    #
            # XXXX is a zero-padded 4-digit index (0000 through 0010).       #
            # We collect the index strings so callers can load specific files #
            # later without hard-coding the count.                            #
            # -------------------------------------------------------------- #
            angles = []
            for filename in sorted(os.listdir(room_path)):
                # re.match anchors at the start of the string, so we look
                # for "rerender_" followed by exactly 4 digits and ".webp"
                match = re.match(r"^rerender_(\d{4})\.webp$", filename)
                if match:
                    # match.group(1) is the 4-digit index, e.g. "0003"
                    angles.append(match.group(1))

            # Only include rooms that actually have at least one rendered angle
            if not angles:
                continue

            # -------------------------------------------------------------- #
            # Step 5: Build the result dictionary for this bedroom room and  #
            # append it to our running list.                                  #
            # -------------------------------------------------------------- #
            bedrooms.append({
                "uuid":      uuid,
                "room_name": room_name,
                "room_path": room_path,
                "angles":    angles,
            })

    return bedrooms


# -------------------------------------------------------------------------- #
# __main__ block: runs when you execute this file directly, e.g.:            #
#   python scanner.py                                                         #
# Useful for quickly verifying the dataset looks correct.                    #
# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Hard-coded path to the raw dataset — change if your layout differs
    DATASET_ROOT = (
        "/Users/karim/Desktop/folders/693-project"
        "/data/raw/3D-FRONT-TEST-RENDER"
    )

    print(f"Scanning dataset at:\n  {DATASET_ROOT}\n")
    results = scan_bedrooms(DATASET_ROOT)

    # ------------------------------------------------------------------ #
    # Summary statistics                                                  #
    # ------------------------------------------------------------------ #
    total_rooms = len(results)
    total_angles = sum(len(r["angles"]) for r in results)

    print(f"Total bedroom rooms found : {total_rooms}")
    print(f"Total angle images found  : {total_angles}")

    if total_rooms > 0:
        avg_angles = total_angles / total_rooms
        print(f"Average angles per room   : {avg_angles:.1f}")

    # Print the first few entries so you can visually sanity-check the output
    print("\nFirst 5 entries:")
    for entry in results[:5]:
        print(
            f"  UUID={entry['uuid'][:8]}...  "
            f"Room={entry['room_name']:<30}  "
            f"Angles={entry['angles']}"
        )
