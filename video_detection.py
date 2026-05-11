import os
import sys
import time
import yaml # Added for loading nutrition data

import cv2
import numpy as np
from ultralytics import YOLO
from IPython.display import display, Video

# input params
model_path = '/content/runs/detect/train/weights/best.pt'
input_video_path = '/content/input_video.mp4'
output_video_path = '/content/output_video.mp4'
min_thresh = 0.5 # Minimum confidence threshold for displaying detected objects
gamma = 0.8 # Added: Gamma correction factor. Values < 1.0 darken, values > 1.0 brighten.

# --- Load Nutrition Data ---
nutrition_data_path = '/content/nutrition_data.yaml'
nutrition_map = {}
if os.path.exists(nutrition_data_path):
    try:
        with open(nutrition_data_path, 'r') as f:
            nutrition_config = yaml.safe_load(f)
        for item in nutrition_config.get('classes', []):
            name = item.get('name')
            nutrition_info = item.get('nutrition', {})
            calories = nutrition_info.get('calories_per_serving')
            protein = nutrition_info.get('protein_grams')
            if name and isinstance(calories, (int, float)) and isinstance(protein, (int, float)):
                nutrition_map[name] = {'calories_per_serving': calories, 'protein_grams': protein}
            else:
                print(f"WARNING: Incomplete or invalid nutrition data for item '{name}' in nutrition_data.yaml. Skipping.")
        print(f"Loaded nutrition data for {len(nutrition_map)} classes.")
    except Exception as e:
        print(f"ERROR: Could not load nutrition data from {nutrition_data_path}: {e}. Nutritional key will not be displayed.")
else:
    print(f"WARNING: Nutrition data file not found at {nutrition_data_path}. Nutritional key will not be displayed.")

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print(f'ERROR: Model path {model_path} is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Open the video file
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {input_video_path}.")
    sys.exit(0)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
# Using 'mp4v' for .mp4 output. If you continue to see brightness issues,
# you might try other codecs, e.g., cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
recorder = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200

print(f"Processing video: {input_video_path}")
print(f"Output video will be saved to: {output_video_path}")

# Begin inference loop
frame_count = 0
while cap.isOpened():
    t_start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break # End of video

    frame_count += 1

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0

    # Initialize variables for nutrition key
    total_calories_frame = 0
    total_protein_frame = 0 # Added: Initialize total protein for the frame
    detected_class_nutrition = {} # Stores {classname: {'calories_per_protein_str': X, 'calories_per_serving': Y, 'protein_grams': Z}}

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

            # Basic example: count the number of objects in the image
            object_count = object_count + 1

            # Process nutrition data for detected object
            if classname in nutrition_map:
                nut_info = nutrition_map[classname]
                calories_per_serving = nut_info.get('calories_per_serving')
                protein_grams = nut_info.get('protein_grams')

                if isinstance(calories_per_serving, (int, float)):
                    total_calories_frame += calories_per_serving
                if isinstance(protein_grams, (int, float)):
                    total_protein_frame += protein_grams # Added: Accumulate total protein

                if classname not in detected_class_nutrition:
                    cal_per_protein_str = 'N/A'
                    if isinstance(calories_per_serving, (int, float)) and isinstance(protein_grams, (int, float)) and protein_grams > 0:
                        cal_per_protein = calories_per_serving / protein_grams
                        cal_per_protein_str = f'{cal_per_protein:.1f}'

                    detected_class_nutrition[classname] = {
                        'calories_per_protein_str': cal_per_protein_str,
                        'calories_per_serving': calories_per_serving,
                        'protein_grams': protein_grams # Added: Store protein grams per serving
                    }

    # Calculate and draw framerate
    cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw framerate
    cv2.putText(frame, f'Objects: {object_count}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects

    # --- Draw Nutrition Key ---
    if nutrition_map: # Only draw key if nutrition data was loaded
        key_lines = []
        key_lines.append(f'Total Calories: {total_calories_frame}')
        key_lines.append(f'Total Protein: {total_protein_frame}g') # Added: Display total protein
        for class_name, nut_data in detected_class_nutrition.items():
            key_lines.append(f'{class_name}: {nut_data["calories_per_serving"]} cal, {nut_data["protein_grams"]}g prot, {nut_data["calories_per_protein_str"]} cal/g_prot') # Updated format

        if key_lines:
            text_color = (255, 255, 255) # White color for text
            background_color = (0, 0, 0) # Black background for readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0 # Increased font size
            font_thickness = 2 # Increased font thickness
            line_height = 40 # Increased line height for larger font

            max_text_width = 0
            for line in key_lines:
                (w, h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                max_text_width = max(max_text_width, w)

            key_width = max_text_width + 40 # Increased padding
            key_height = len(key_lines) * line_height + 20 # Increased padding

            key_x = frame_width - key_width - 20 # Increased offset from right edge
            key_y = 20 # Increased offset from top edge

            # Draw black background rectangle for the key
            cv2.rectangle(frame, (key_x, key_y), (key_x + key_width, key_y + key_height), background_color, cv2.FILLED)

            # Draw key text
            current_y = key_y + line_height // 2 + 10 # Adjusted initial y for text
            for line in key_lines:
                cv2.putText(frame, line, (key_x + 10, current_y), font, font_scale, text_color, font_thickness)
                current_y += line_height

    # Apply gamma correction before writing the frame
    if gamma != 1.0:
        # Normalize frame values to 0-1, apply gamma, then scale back to 0-255
        gamma_corrected_frame = np.array(255 * (frame / 255.0)**gamma, dtype=np.uint8)
        recorder.write(gamma_corrected_frame)
    else:
        recorder.write(frame)

    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        _ = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)

    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames... Current FPS: {avg_frame_rate:.2f}")

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
cap.release()
recorder.release()
print("Video processing complete. Output saved to /content/output_video.mp4")