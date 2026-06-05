import pyrealsense2 as rs
import cv2
import numpy as np

print("pyrealsense2 ok")
print(f"opencv version: {cv2.__version__}")
print(f"numpy version: {np.__version__}")

ctx = rs.context()
devices = ctx.query_devices()

for i, dev in enumerate(devices):
    print(f"\ndevice {i}")
    print(f"Name: {dev.get_info(rs.camera_info.name)}")
    print(f"Serial: {dev.get_info(rs.camera_info.serial_number)}")
    print(f"Firmware: {dev.get_info(rs.camera_info.firmware_version)}")