import cv2

def find_aruco_dict(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # --- THE FIX: ADD A WHITE BORDER (QUIET ZONE) ---
    # ArUco detectors need a white margin around the black border to find the contour.
    border_size = 50
    image = cv2.copyMakeBorder(
        image, 
        top=border_size, 
        bottom=border_size, 
        left=border_size, 
        right=border_size, 
        borderType=cv2.BORDER_CONSTANT, 
        value=[255, 255, 255] # White
    )

    # A dictionary mapping string names to OpenCV ArUco dictionary flags
    aruco_dicts = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
    }

    print("Scanning for marker...")
    
    # Loop through each dictionary and try to detect the marker
    for dict_name, dict_flag in aruco_dicts.items():
        dictionary = cv2.aruco.getPredefinedDictionary(dict_flag)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        
        corners, ids, rejected = detector.detectMarkers(image)
        
        if ids is not None:
            print(f"\n✅ Match Found!")
            print(f"--> Dictionary: {dict_name}")
            print(f"--> Marker ID: {ids[0][0]}")
            return # Exit after finding the first match

    print("\n❌ No standard ArUco marker detected in the image.")

if __name__ == "__main__":
    find_aruco_dict("aruco_markers/aruco2.png")