import os
from datetime import datetime
from pathlib import Path

import json

import pyrealsense2 as rs
import cv2
import numpy as np
import open3d as o3d



WIDTH = 640
HEIGHT = 480
ROI_SCALE = 0.6
FPS = 30 

MIN_DEPTH_M = 0.07
MAX_DEPTH_M = 0.50

CAMERA_PRESETS = {
    "D405": {
        "min_depth_m": 0.07,
        "max_depth_m": 0.50,
        "roi_scale": 0.60,
    },
    "D455F": {
        "min_depth_m": 0.30,
        "max_depth_m": 2.00,
        "roi_scale": 0.60,
    },
    "DEFAULT": {
        "min_depth_m": 0.10,
        "max_depth_m": 1.50,
        "roi_scale": 0.60,
    },
}

# localCaptures: 
#   Temporary local testing data.
#   main_capture.py writes here by default
#
# captures:
#   Clean / selected data only
#   Move useful session here manually before cloud/server upload.


LOCAL_SAVE_DIR = "localCaptures"
CLOUD_SAVE_DIR = "captures"

SAVE_DIR = LOCAL_SAVE_DIR

os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
os.makedirs(CLOUD_SAVE_DIR, exist_ok = True)

def configure_save_dirs(camera_preset_name):
    global SAVE_DIR

    local_camera_dir = os.path.join(LOCAL_SAVE_DIR, camera_preset_name)
    cloud_camera_dir = os.path.join(CLOUD_SAVE_DIR, camera_preset_name)

    os.makedirs(local_camera_dir, exist_ok=True)
    os.makedirs(cloud_camera_dir, exist_ok=True)

    SAVE_DIR = local_camera_dir

    print(f"\n[SAVE Directory]")
    print(f"Local save dir: {local_camera_dir}")
    print(f"Cloud save dir: {cloud_camera_dir}")

    return local_camera_dir, cloud_camera_dir

def detect_camera_name(profile):
    device = profile.get_device()

    try:
        name = device.get_info(rs.camera_info.name)
    except Exception:
        name = "Unknown"

    try:
        serial = device.get_info(rs.camera_info.serial_number)
    except Exception:
        serial = "Unknown"

    print("\n[Device Info]")
    print(f"Name  : {name}")
    print(f"Serial: {serial}")

    return name


def apply_camera_preset(camera_name):
    global MIN_DEPTH_M, MAX_DEPTH_M, ROI_SCALE

    name_upper = camera_name.upper()

    if "D405" in name_upper:
        preset = CAMERA_PRESETS["D405"]
        preset_name = "D405"
    elif "D455" in name_upper:
        preset = CAMERA_PRESETS["D455F"]
        preset_name = "D455F"
    else:
        preset = CAMERA_PRESETS["DEFAULT"]
        preset_name = "DEFAULT"

    MIN_DEPTH_M = preset["min_depth_m"]
    MAX_DEPTH_M = preset["max_depth_m"]
    ROI_SCALE = preset["roi_scale"]

    print("\n[Camera Preset]")
    print(f"Preset     : {preset_name}")
    print(f"Min depth  : {MIN_DEPTH_M} m")
    print(f"Max depth  : {MAX_DEPTH_M} m")
    print(f"ROI scale  : {ROI_SCALE}")

    return preset_name


def make_depth_colormap(depth_image, depth_scale):
    
    depth_m = depth_image * depth_scale

    depth_vis = np.clip(depth_m, MIN_DEPTH_M, MAX_DEPTH_M)
    depth_vis = ((depth_vis - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M)* 255).astype(np.uint8)

    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # depth == 0 means invalid 
    depth_colormap[depth_image == 0] = 0

    return depth_colormap

def make_depth_grayscale(depth_image, depth_scale):

    depth_m = depth_image * depth_scale
    
    depth_vis = np.clip(depth_m, MIN_DEPTH_M, MAX_DEPTH_M)
    depth_gray = ((depth_vis - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M) * 255).astype(np.uint8)

    depth_gray[depth_image ==0] = 0

    return depth_gray

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

def crop_center_roi(color_image, depth_image, roi_scale= 0.6):
    h, w = depth_image.shape

    x1, y1, x2, y2 = get_center_roi_bounds(w, h, roi_scale)

    color_roi = np.ascontiguousarray(color_image[y1:y2, x1:x2])
    depth_roi = np.ascontiguousarray(depth_image[y1:y2, x1:x2])

    print(f"\n[ROI]")
    print(f"x: {x1} -> {x2}")
    print(f"y: {y1} -> {y2}")
    print(f"size: {x2 - x1} x {y2 - y1}")

    return color_roi, depth_roi, x1, y1, x2, y2

def get_center_roi_bounds(image_width, image_height, roi_scale=0.6):
    roi_w = int(image_width * roi_scale)
    roi_h = int(image_height * roi_scale)

    x1 = (image_width - roi_w) // 2
    y1 = (image_height - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    return x1, y1, x2, y2

def create_roi_intrinsic(color_frame, x_offset, y_offset, roi_width, roi_height):

    intr = color_frame.profile.as_video_stream_profile().intrinsics

    roi_fx = intr.fx
    roi_fy = intr.fy
    roi_cx = intr.ppx - x_offset
    roi_cy = intr.ppy - y_offset

    print(f"\n[ROI Camera Intrinsic]")
    print(f"width : {roi_width}")
    print(f"height: {roi_height}")
    print(f"fx    : {roi_fx}")
    print(f"fy    : {roi_fy}")
    print(f"cx    : {roi_cx}")
    print(f"cy    : {roi_cy}")

    return o3d.camera.PinholeCameraIntrinsic(
        roi_width,
        roi_height,
        roi_fx,
        roi_fy,
        roi_cx,
        roi_cy
    )


def create_point_cloud(color_image_bgr, depth_image_raw, depth_scale, intrinsic):

    color_image_bgr = np.ascontiguousarray(color_image_bgr)
    depth_image_raw = np.ascontiguousarray(depth_image_raw)

    color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
    color_image_rgb = np.ascontiguousarray(color_image_rgb)

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
    depth_gray_path = os.path.join(SAVE_DIR, f"depth_gray_{timestamp}.png")

    cv2.imwrite(color_path, color_image_bgr)

    depth_mm = (depth_image_raw * depth_scale * 1000.0).astype(np.uint16)
    cv2.imwrite(depth_mm_path, depth_mm)

    depth_vis = make_depth_colormap(depth_image_raw, depth_scale)
    cv2.imwrite(depth_vis_path, depth_vis)

    depth_gray = make_depth_grayscale(depth_image_raw, depth_scale)
    cv2.imwrite(depth_gray_path, depth_gray)

    print(f"[SAVED] {color_path}")
    print(f"[SAVED] {depth_mm_path}")
    print(f"[SAVED] {depth_vis_path}")
    print(f"[SAVED] {depth_gray_path}")

    return {
        "color" : os.path.basename(color_path),
        "depth_mm" : os.path.basename(depth_mm_path),
        "depth_vis" : os.path.basename(depth_vis_path),
        "depth_gray" : os.path.basename(depth_gray_path)
    }

def clean_point_cloud(pcd):
  ##  Step 1:
  ##      voxel_down_sample()
  ##      Reduce point density.

  ##  Step 2:
  ##      remove_statistical_outlier()
  ##      Remove isolated noisy points.
  

    raw_count = len(pcd.points)
    print("\n[Clean Point Cloud]")
    print(f"Raw points: {raw_count}")

    # 1 mm voxel size.
    pcd_down = pcd.voxel_down_sample(voxel_size=0.001)

    down_count = len(pcd_down.points)
    print(f"After voxel downsample: {down_count}")

    # Remove isolated outlier points.
    pcd_clean, indices = pcd_down.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

    clean_count = len(pcd_clean.points)
    print(f"After outlier removal: {clean_count}")

    return pcd_clean

def compute_bounding_box(pcd):
    print(f"\nBounding box")

    if len(pcd.points) < 10:
        print(f"[WARN] Not enough points for creating bounding box.")
        return None, None
    
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()

    width_m = extent[0]
    height_m = extent[1]
    depth_m = extent[2]

    measurement = {
        "width_mm" : float(width_m * 1000.0),
        "height_mm" : float(height_m * 1000.0),
        "depth_mm" : float(depth_m * 1000.0),
        "point_count" : len(pcd.points),
    }

    print(f"width: {measurement['width_mm']:.2f} mm")
    print(f"height: {measurement['height_mm']:.2f} mm")
    print(f"depth: {measurement['depth_mm']:.2f} mm")
    print(f"points: {measurement['point_count']}")

    return bbox, measurement

def remove_plane(pcd, distance_threshold=0.003, ransac_n=3, num_iterations=1000):
    print("\n[Plane Removal]")

    if len(pcd.points) < 100:
        print("[WARN] Not enough points for plane removal.")
        return pcd, None

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    a, b, c, d = plane_model

    print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"Plane inliers: {len(inliers)}")

    plane_cloud = pcd.select_by_index(inliers)
    object_cloud = pcd.select_by_index(inliers, invert=True)

    print(f"Object points after plane removal: {len(object_cloud.points)}")


    return object_cloud, plane_cloud

def save_point_cloud(pcd, timestamp, prefix="pointcloud"):
    ply_path = os.path.join(SAVE_DIR, f"{prefix}_{timestamp}.ply")

    ok = o3d.io.write_point_cloud(ply_path, pcd)

    if ok:
        print(f"[SAVED] {ply_path}")
    else:
        print("[ERROR] Failed to save point cloud.")

    return ply_path

def save_measurement_json(measurement, timestamp):
    json_path = os.path.join(SAVE_DIR, f"measurement_{timestamp}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(measurement, f, indent= 2)
    
    print(f"[SAVED] {json_path}")
    return json_path

def save_metadata_json(
    timestamp, camera_name, 
    camera_preset_name, roi_info,
    measurement, files
):
    metadata = {
        "timestamp": timestamp,
        "camera" : {
            "name": camera_name,
            "preset": camera_preset_name,
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "min_depth_m": MIN_DEPTH_M,
            "max_depth_m": MAX_DEPTH_M
        },
        "roi": roi_info,
        "measurement": measurement,
        "files":files,
    }
    json_path = os.path.join(SAVE_DIR, f"metadata_{timestamp}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SAVED] {json_path}")
    return json_path

def update_capture_index(timestamp, metadata_filename):
    index_path = os.path.join(SAVE_DIR, "index.json")

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            try:
                index_data = json.load(f)
            except json.JSONDecodeError:
                index_data = []
    else:
        index_data = []
    
    new_entry = {
        "timestamp": timestamp,
        "metadata": metadata_filename,
    }

    index_data.append(new_entry)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    print(f"[UPDATED] {index_path}")
    
    return index_path

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS) # For openCV compatibility

    print("[INFO] Starting RealSense pipeline...")
    
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    camera_name = detect_camera_name(profile)
    camera_preset_name = apply_camera_preset(camera_name)
    configure_save_dirs(camera_preset_name)


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

            depth_frame = align_frames.get_depth_frame()
            color_frame = align_frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("[WARN] Missing frame")
                continue
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = make_depth_colormap(depth_image, depth_scale)
            depth_gray = make_depth_grayscale(depth_image, depth_scale)

            # Convert grayscale depth to 3-channel BGR
            # So it can be stacked witg RGB depth color
            depth_gray_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
            
            preview_color = color_image.copy()
            preview_depth = depth_colormap.copy()
            preview_depth_gray = depth_gray_bgr.copy()

            x1, y1, x2, y2 = get_center_roi_bounds(WIDTH, HEIGHT, ROI_SCALE)

            cv2.putText(preview_color, "RGB", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(preview_depth, "DEPTH", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(preview_depth_gray, "DEPTH GRAY", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


            cv2.rectangle(preview_color, (x1, y1), (x2, y2), (0, 255, 0), 2)          
            cv2.rectangle(preview_depth, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(preview_depth_gray, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            combined = np.hstack((preview_color, preview_depth, preview_depth_gray))
            cv2.imshow("RealSense RGB | Depth Color | Depth Gray", combined) 

            key = cv2.waitKey(1)

            if key == 27:
                break
            if key == 32:
                print(f"\n[INFO] Creating point cloud from center ROI...")

                color_roi, depth_roi, x1, y1, x2, y2 = crop_center_roi(
                    color_image=color_image,
                    depth_image=depth_image,
                    roi_scale=ROI_SCALE
                )

                roi_intrinsic = create_roi_intrinsic(
                    color_frame=color_frame,
                    x_offset=x1,
                    y_offset=y1,
                    roi_width=depth_roi.shape[1],
                    roi_height=depth_roi.shape[0]
                )

                pcd = create_point_cloud(
                    color_image_bgr=color_roi,
                    depth_image_raw=depth_roi,
                    depth_scale=depth_scale,
                    intrinsic=roi_intrinsic
                )

                print(f"\n[INFO] Number of ROI points: {len(pcd.points)}")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                image_files = save_debug_images(
                    color_image_bgr=color_roi,
                    depth_image_raw=depth_roi,
                    depth_scale=depth_scale,
                    timestamp=timestamp
                )

                pcd_roi_path = save_point_cloud(pcd, timestamp, prefix="pointcloud_roi")

                pcd_clean = clean_point_cloud(pcd)
                pcd_clean_path = save_point_cloud(pcd_clean, timestamp, prefix="pointcloud_roi_clean")

                pcd_object, pcd_plane = remove_plane(pcd_clean)
                pcd_object_path = save_point_cloud(pcd_object, timestamp, prefix="pointcloud_object")

                pcd_plane_path = None
                if pcd_plane is not None:
                    pcd_plane_path = save_point_cloud(pcd_plane, timestamp, prefix="pointcloud_plane")

                bbox, measurement = compute_bounding_box(pcd_object)

                if bbox is not None:
                    bbox.color= (0, 1, 0)

                roi_info = {
                    "scale" : ROI_SCALE,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                }

                files = {
                    **image_files,
                    "pointcloud_roi": os.path.basename(pcd_roi_path),
                    "pointcloud_clean": os.path.basename(pcd_clean_path),
                    "pointcloud_object": os.path.basename(pcd_object_path),
                    "pointcloud_plane": os.path.basename(pcd_plane_path) if pcd_plane_path else None,
                }

                if measurement is not None:
                    save_measurement_json(measurement, timestamp)

                    metadata_path = save_metadata_json(
                        timestamp =timestamp,
                        camera_name = camera_name,
                        camera_preset_name = camera_preset_name,
                        roi_info = roi_info,
                        measurement = measurement,
                        files = files,
                    )

                    update_capture_index(
                        timestamp = timestamp,
                        metadata_filename = os.path.basename(metadata_path)
                    )

                print("[INFO] Opening Open3D Viewer...")

                if bbox is not None:
                    o3d.visualization.draw_geometries([pcd_object, bbox])
                else:
                    o3d.visualization.draw_geometries([pcd_object])
                
    finally:
        print("[INFO] Stopping pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()