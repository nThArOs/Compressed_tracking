import cv2
import os
import numpy as np
import configparser

# Root directory of the dataset
root_dir = "/home/modesto/PycharmProjects/compressed_tracking/datasets/MOT17/train/"

# Directory where to save the cropped person data
save_dir = "/home/modesto/PycharmProjects/compressed_tracking/datasets/MOT17-reid"

# Loop over the sequences
for directory in os.listdir(root_dir):
    # Load the ground truth file
    print(directory)
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
        print(frame_num)
        frame_num = int(frame_num)
        person_id = int(person_id)

        
        if frame_num % 12 == 1:
            continue

        # Load the image file for this frame
        img_file = os.path.join(root_dir, sequence, "img1", f"{frame_num:06}.png")
        try:
            img = cv2.imread(img_file)
        except:
            print("skip frame")
            continue  # skip this frame if the image file does not exist

        # Handle case where image is None
        if img is None:
            continue

        # Get the actual dimensions of the image
        actual_height, actual_width, _ = img.shape

        # Compute scale factors
        scale_y = actual_height / original_height
        scale_x = actual_width / original_width
       # print(scale_y,scale_x)
        # Scale the bounding box coordinates
        x, w = int(x * scale_x), int(w * scale_x)
        y, h = int(y * scale_y), int(h * scale_y)
        

        # Crop the image
        x = max(0, x)
        y = max(0, y)
        w = min(w, actual_width - x)
        h = min(h, actual_height - y)

        # If the bounding box has become zero-sized due to corrections, skip it
        if w <= 0 or h <= 0:
            continue

        cropped_img = img[y:y+h, x:x+w]

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


