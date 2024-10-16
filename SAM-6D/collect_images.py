## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# Import necessary libraries
import pyrealsense2 as rs
import numpy as np
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream color and depth streams
config = rs.config()

# Get device product line to set the appropriate resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Check for RGB camera availability
found_rgb = any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device.sensors)
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# Enable color and depth streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Create an alignment object to align depth to color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Global variable to store the latest frames
latest_color_image = None
latest_depth_image = None
click_count = 0

def on_mouse(event, x, y, flags, param):
    """Save the clicked coordinates and corresponding images to files."""
    global click_count, latest_color_image, latest_depth_image

    if event == cv2.EVENT_LBUTTONDOWN:
        # Increment the click count for unique filenames
        click_count += 1

        # Save the clicked coordinates to a text file
        coords = f"{x}, {y}\n"
        print(f"Clicked coordinates: {coords.strip()}")
        with open("clicked_coordinates.txt", "a") as f:
            f.write(coords)

        # Save the current images if they are available
        if latest_color_image is not None and latest_depth_image is not None:
            color_path = f'aligned_color_click{click_count}.png'
            depth_path = f'aligned_depth_click{click_count}.png'
            cv2.imwrite(color_path, latest_color_image)
            cv2.imwrite(depth_path, latest_depth_image)
            print(f'Saved images: {color_path}, {depth_path}')

# Set the mouse callback function
cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Align Example', on_mouse)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to the color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate frames
        if not aligned_depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        latest_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        latest_color_image = np.asanyarray(color_frame.get_data())

        # Render combined image with color map for depth
        images = np.hstack((
            latest_color_image,
            cv2.applyColorMap(cv2.convertScaleAbs(latest_depth_image, alpha=0.03), cv2.COLORMAP_JET)
        ))

        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)

        # Exit on 'q' or Esc key press
        if key & 0xFF == ord('q') or key == 27:
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
