import cv2
import os
import numpy as np
import configparser

# Root directory of the dataset
root_dir = "/home/modesto/PycharmProjects/compressed_tracking/datasets/MOT17/train/"

# Directory where to save the cropped person data
save_dir = "/home/modesto/PycharmProjects/compressed_tracking/datasets/MOT17-reid5D"

# Loop over the sequences
for directory in os.listdir(root_dir):
    # Load the ground truth file
    if not directory.startswith("MOT17"):
        continue

    # This directory is a MOT17 sequence
    sequence = directory

    # Parse the seqinfo.ini file to get original image dimensions
    config = configparser.ConfigParser()
    config.read(os.path.join(root_dir, sequence, "seqinfo.ini"))
    original_width = int(config["Sequence"]["imWidth"])
    original_height = int(config["Sequence"]["imHeight"])

    gt_file = os.path.join(root_dir, sequence, "gt", "gt.txt")
    gt_data = np.loadtxt(gt_file, delimiter=',')

    # Dictionary to hold person data
    person_data = {}

    # Loop over the ground truth data
    for frame_num, person_id, x, y, w, h, _, _, _ in gt_data:
        # Convert frame_num and person_id to integers
        frame_num = int(frame_num)
        person_id = int(person_id)

        if frame_num % 12 == 1:
            continue

        # Load the image file for this frame
        img_file = os.path.join(root_dir, sequence, "residual", f"{frame_num:06}.png")
        try:
            img = cv2.imread(img_file)
        except:
            continue  # skip this frame if the image file does not exist

        # Handle case where image is None
        if img is None:
            continue
        #   print("frame_num",frame_num)
        #   print(f"Original bounding box values: x={x}, y={y}, w={w}, h={h}")

        # Get the actual dimensions of the image
        actual_height, actual_width, _ = img.shape

        # Compute scale factors
        scale_y = actual_height / original_height
        scale_x = actual_width / original_width

        # Scale the bounding box coordinates
        x, w = int(x * scale_x), int(w * scale_x)
        y, h = int(y * scale_y), int(h * scale_y)

        #  print(f"Scaled bounding box values: x={x}, y={y}, w={w}, h={h}")

        # Crop the image
        x = max(0, x)
        y = max(0, y)
        w = min(w, actual_width - x)
        h = min(h, actual_height - y)

        # If the bounding box has become zero-sized due to corrections, skip it
        if w <= 0 or h <= 0:
            continue

        cropped_img = img[y:y + h, x:x + w]

        # Handle case where cropped_img is empty
        if cropped_img.size == 0:
            print(f"Empty cropped image for frame: {frame_num}{person_id}")
            print(f"Bounding box coordinates: x={x}, y={y}, w={w}, h={h}")
            print(f"Original image size: {img.shape}")

            continue

        # Create a directory to hold this person's data if it doesn't exist yet
        if person_id not in person_data:
            person_data[person_id] = os.path.join(save_dir, sequence, str(person_id))
            os.makedirs(person_data[person_id], exist_ok=True)

        # Save the cropped image
        cv2.imwrite(os.path.join(person_data[person_id], f"{frame_num:06}.png"), cropped_img)

        # Load the data file for this frame
        data_file = os.path.join(root_dir, sequence, "mv", f"{frame_num:06}.npy")
        try:
            data = np.load(data_file)
        except FileNotFoundError:
            continue  # skip this frame if the .npy file does not exist

        # Crop the data
        cropped_data = data[y:y + h, x:x + w, :]

        # Save the cropped data
        np.save(os.path.join(person_data[person_id], f"{frame_num:06}.npy"), cropped_data)

