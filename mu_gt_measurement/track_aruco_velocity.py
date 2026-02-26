import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import csv
from datetime import datetime

# --- CONFIGURATION ---
MARKER_SIZE_METERS = 0.066  
ARUCO_DICT = cv2.aruco.DICT_4X4_1000 

def main():
    # Setup Output Directories and Files
    os.makedirs("videos", exist_ok=True)
    os.makedirs("aruco_velocity", exist_ok=True)

    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filepath = os.path.join("videos", f"tracking_{session_time}.mp4")
    csv_filepath = os.path.join("aruco_velocity", f"velocity_{session_time}.csv")

    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter(video_filepath, fourcc, 30.0, (640, 480))
    video_writer = cv2.VideoWriter(video_filepath, fourcc, 90.0, (480, 270))

    # Initialize CSV Data Logger
    csv_file = open(csv_filepath, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Added Px, Py, and Pz to the header
    csv_writer.writerow(['Timestamp', 'Px', 'Py', 'Pz', 'Speed_m_s', 'Vx', 'Vy', 'Vz'])

    # 1. Initialize RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 90)
    

    profile = pipeline.start(config)

    # Get the sensor handle for the RGB camera
    device = profile.get_device()
    color_sensor = device.first_color_sensor()

    # 1. Disable Auto Exposure
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)

    # 2. Set Manual Exposure (Value is in microseconds)
    # Start with 1000. If the image is too dark, increase to 2000.
    # If the marker still blurs, decrease to 500 (requires very bright room light).
    # color_sensor.set_option(rs.option.exposure, 500)
    color_sensor.set_option(rs.option.exposure, 100)

    # 3. Set Gain (Higher gain makes it b righter but noisier)
    # Values range from 0 to 128. Try 64.
    color_sensor.set_option(rs.option.gain, 64)

    print("✅ Manual Exposure Set to 1000us. Ensure your work area is BRIGHTLY lit.")
    
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                              [0, intrinsics.fy, intrinsics.ppy],
                              [0, 0, 1]])
    dist_coeffs = np.array(intrinsics.coeffs)

    # 2. Setup ArUco Detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    prev_tvec = None
    prev_time = None

    print(f"Recording video to: {video_filepath}")
    print(f"Logging data to: {csv_filepath}")
    print("Press 'q' to quit...")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            color_image = np.asanyarray(color_frame.get_data())
            current_time = time.time()

            # --- NEW: ALWAYS DRAW CURRENT TIME (0.01s precision) ---
            live_time_str = datetime.now().strftime("%H:%M:%S.%f")[:-4]
            # CHANGED: Moved from 460 to 250 so it fits on a 270p screen!
            cv2.putText(color_image, f"Time: {live_time_str}", (10, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
            # -------------------------------------------------------

            # 3. Detect ArUco markers
            corners, ids, rejected = detector.detectMarkers(color_image)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_METERS, camera_matrix, dist_coeffs)

                for i in range(len(ids)):
                    # tvec is the 3D position [X, Y, Z] in meters
                    tvec = tvecs[i][0]
                    px, py, pz = tvec[0], tvec[1], tvec[2]
                    
                    cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvecs[i], tvec, MARKER_SIZE_METERS / 2)

                    # 4. Calculate Velocity & Log Data
                    if prev_tvec is not None and prev_time is not None:
                        dt = current_time - prev_time
                        if dt > 0:
                            velocity_vector = (tvec - prev_tvec) / dt
                            # Calculate speed ONLY on the flat table surface (Left/Right X and Depth Z)
                            speed = np.linalg.norm([velocity_vector[0], velocity_vector[2]])

                            velocity_text = f"Speed: {speed:.3f} m/s"
                            cv2.putText(color_image, velocity_text, (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            
                            coords_text = f"P_x:{px:.2f} P_y:{py:.2f} P_z:{pz:.2f}"
                            cv2.putText(color_image, coords_text, (10, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

                            # Saving Px, Py, and Pz directly into the CSV
                            csv_writer.writerow([
                                f"{current_time:.4f}", 
                                f"{px:.4f}", 
                                f"{py:.4f}", 
                                f"{pz:.4f}", 
                                f"{speed:.4f}", 
                                f"{velocity_vector[0]:.4f}", 
                                f"{velocity_vector[1]:.4f}", 
                                f"{velocity_vector[2]:.4f}"
                            ])

                    prev_tvec = tvec
                    prev_time = current_time

            else:
                prev_tvec = None
                prev_time = None

            video_writer.write(color_image)
            cv2.imshow('RealSense ArUco Tracker (Position + Velocity)', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        video_writer.release()
        csv_file.close()
        cv2.destroyAllWindows()
        print("Data saved successfully. Exiting.")

if __name__ == "__main__":
    main()