import pyrealsense2 as rs
import cv2
import numpy as np

WIDTH = 640
HEIGHT = 480
FPS = 30 

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    print("[INFO] Starting RealSense pipeline...")
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print(f"[INFO] Depth scale: {depth_scale}")
    print(f"[INFO] Press ESC to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("[WARN] Missing frame")
                continue
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_frame = np.asanyarray(color_frame.get_data())

            depth_m = depth_image * depth_scale

            depth_vis = np.clip(depth_m, 0.07, 0.50)
            depth_vis = ((depth_vis - 0.07) / (0.50 - 0.07) * 255).astype(np.uint8)

            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            combined = np.hstack((color_frame, depth_colormap))

            cv2.imshow("D405 RGB | Depth", combined)

            key = cv2.waitKey(1)
            if key == 27:
                break;
    
    finally:
        print("[INFO] Stopping pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()