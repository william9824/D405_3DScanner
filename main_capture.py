import os
from datetime import datetime

import pyrealsense2 as rs
import cv2
import numpy as np
import open3d as o3d


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

def get_open3d_intrinsic(color_frame):
    intr = color_frame.profile.as_video_stream_profile().intrinsics

    print("\n[Camera Intrinsic]")
    print("width :", intr.width)
    print("height:", intr.height)
    print("fx    :", intr.fx)
    print("fy    :", intr.fy)
    print("cx    :", intr.ppx)
    print("cy    :", intr.ppy)

    return o3d.camera.PinholeCameraIntrinsic(
        intr.width,
        intr.height,
        intr.fx,
        intr.fy,
        intr.ppx,
        intr.ppy
    )


def create_point_cloud(color_image_bgr, depth_image_raw, depth_scale, intrinsic):
    color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)

    color_o3d = o3d.geometry.Image(color_image_rgb)
    depth_o3d = o3d.geometry.Image(depth_image_raw)

    open3d_depth_scale = 1.0 / depth_scale

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        depth_scale=open3d_depth_scale,
        depth_trunc=MAX_DEPTH_M,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=rgbd,
        intrinsic=intrinsic
    )

    # Flip for easier viewing in Open3D
    pcd.transform([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ])

    return pcd

def save_debug_images(color_image_bgr, depth_image_raw, depth_scale, timestamp):
    color_path = os.path.join(SAVE_DIR, f"color_{timestamp}.png")
    depth_mm_path = os.path.join(SAVE_DIR, f"depth_mm_{timestamp}.png")
    depth_vis_path = os.path.join(SAVE_DIR, f"depth_vis_{timestamp}.png")

    cv2.imwrite(color_path, color_image_bgr)

    depth_mm = (depth_image_raw * depth_scale * 1000.0).astype(np.uint16)
    cv2.imwrite(depth_mm_path, depth_mm)

    depth_vis = make_depth_colormap(depth_image_raw, depth_scale)
    cv2.imwrite(depth_vis_path, depth_vis)

    print(f"[SAVED] {color_path}")
    print(f"[SAVED] {depth_mm_path}")
    print(f"[SAVED] {depth_vis_path}")

def save_point_cloud(pcd, timestamp):
    ply_path = os.path.join(SAVE_DIR, f"pointcloud_{timestamp}.ply")

    ok = o3d.io.write_point_cloud(ply_path, pcd)

    if ok:
        print(f"[SAVED] {ply_path}")
    else:
        print("[ERROR] Failed to save point cloud.")

    return ply_path


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS) # For openCV compatibility

    print("[INFO] Starting RealSense pipeline...")
    
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print(f"[INFO] Depth scale: {depth_scale}")
    print(f"[INFO] Press ESC to quit.")
    print(f"[INFO] Press SPACE to create point cloud.")

    for _ in range(30):
        pipeline.wait_for_frames()

    try:
        while True:
            frames = pipeline.wait_for_frames()

            align_frames = align.process(frames)

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
                print(f"\n[INFO] Creating poing cloud...")

                intrinsic = get_open3d_intrinsic(color_frame)

                pcd = create_point_cloud(
                    color_image_bgr = color_image,
                    depth_image_raw = depth_image,
                    depth_scale = depth_scale,
                    intrinsic = intrinsic
                )

                print(f"\n[INFO] Number of points: {len(pcd.points)}")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                save_debug_images(
                    color_image_bgr = color_image,
                    depth_image_raw = depth_image,
                    depth_scale = depth_scale,
                    timestamp = timestamp
                )

                save_point_cloud(pcd, timestamp)

                print("[INFO] Opening Opend3D Viewer...")
                o3d.visualization.draw_geometries([pcd])
                
    finally:
        print("[INFO] Stopping pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()