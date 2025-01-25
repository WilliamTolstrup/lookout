import cv2
import numpy as np

# Load image
image = cv2.imread('/home/william/Pictures/Screenshots/white-aruco-2.jpg')
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters_create()

corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
if ids is not None:
    print(f"Detected marker IDs: {ids}")
    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
else:
    print("No markers detected")

cv2.imshow('Markers', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
