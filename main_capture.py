import os
from datetime import datetime

import pyrealsense2 as rs
import cv2
import numpy as np

WIDTH = 640
HEIGHT = 480
FPS = 30 

MIN_DEPTH_M = 0.07
MAX_DEPTH_M = 0.50

SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok = True)

def make_depth_colormap(depth_image, depth_scale):
    
    depth_m = depth_image * depth_scale

    depth_vis = np.clip(depth_m, MIN_DEPTH_M, MAX_DEPTH_M)
    depth_vis = ((depth_vis - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M)* 255).astype(np.uint8)

    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # depth == 0 means invalid 
    depth_colormap[depth_image == 0] = 0

    return depth_colormap

def save_capture(color_image, depth_image, depth_scale):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    color_path = os.path.join(SAVE_DIR, f"color_{timestamp}.png")
    depth_mm_path = os.path.join(SAVE_DIR, f"depth_mm_{timestamp}.png")
    depth_vis_path = os.path.join(SAVE_DIR, f"depth_vis_{timestamp}.png")

    cv2.imwrite(color_path, color_image)

    # Save as uint16 PNG so we keep real distance data.
    depth_mm = (depth_image * depth_scale * 1000.0).astype(np.uint16)

    cv2.imwrite(depth_mm_path, depth_mm)
    depth_colormap = make_depth_colormap(depth_image, depth_scale)
    cv2.imwrite(depth_vis_path, depth_colormap)

    print("\n[SAVED]")
    print("Color image:       ", color_path)
    print("Depth image 16-bit:", depth_mm_path)
    print("Depth preview:     ", depth_vis_path)

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS) # For openCV compatibility

    print("[INFO] Starting RealSense pipeline...")
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print(f"[INFO] Depth scale: {depth_scale}")
    print(f"[INFO] Press ESC to quit.")
    print(f"[INFO] Press SPACE to capture.")

    for _ in range(30):
        pipeline.wait_for_frames()

    try:
        while True:
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("[WARN] Missing frame")
                continue
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = make_depth_colormap(depth_image, depth_scale)

            combined = np.hstack((color_image, depth_colormap))
            cv2.imshow("D405 RGB | Depth", combined) 

            key = cv2.waitKey(1)

            if key == 27:
                break;
            if key == 32:
                save_capture(color_image, depth_image, depth_scale)
                
    finally:
        print("[INFO] Stopping pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()