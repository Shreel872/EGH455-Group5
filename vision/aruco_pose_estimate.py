
import cv2
import depthai as dai
import numpy as np

MARKER_SIZE = 0.05  # Marker side length in metres
WIDTH, HEIGHT = 1280, 720
FPS = 30

# Try common ArUco dictionaries
DICTIONARIES = {
    "4X4_1000": cv2.aruco.DICT_4X4_1000,
    "5X5_1000": cv2.aruco.DICT_5X5_1000,
    "6X6_1000": cv2.aruco.DICT_6X6_1000,
    "7x7_1000": cv2.aruco.DICT_7X7_1000,
    "ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "MIP_36h12": cv2.aruco.DICT_ARUCO_MIP_36h12,
}

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

detectors = {
    name: cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dictionary), params
    )
    for name, dictionary in DICTIONARIES.items()
}

# 3D marker corners, matching OpenCV's corner order
s = MARKER_SIZE / 2
object_points = np.array([
    [-s,  s, 0],
    [ s,  s, 0],
    [ s, -s, 0],
    [-s, -s, 0]
], dtype=np.float32)


def rotation_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_A
    )

    rgb = cam.requestOutput(
        size=(WIDTH, HEIGHT),
        type=dai.ImgFrame.Type.BGR888i,
        resizeMode=dai.ImgResizeMode.STRETCH,
        fps=FPS,
        enableUndistortion=False
    )

    queue = rgb.createOutputQueue(maxSize=4, blocking=False)

    calibration = np.load("camera_calibration.npz")

    camera_matrix = calibration["camera_matrix"]
    dist_coeffs = calibration["dist_coeffs"]

    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:", dist_coeffs)

    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:", dist_coeffs)

    pipeline.start()

    while pipeline.isRunning():
        frame = queue.get().getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        count = 0
        rejected_count = 0

        for name, detector in detectors.items():
            corners, ids, rejected = detector.detectMarkers(gray)
            rejected_count += len(rejected)

            if ids is None:
                continue

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for marker_id, marker_corners in zip(ids.flatten(), corners):
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    marker_corners.reshape(4, 2),
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                # Centre of detected marker in image pixels
                u, v = marker_corners.reshape(4, 2).mean(axis=0)

                # Approximate X and Y using estimated Z
                z = float(tvec[2, 0])

                # Correct pixel coordinates for lens distortion
                point = np.array([[[u, v]]], dtype=np.float64)
                normalized = cv2.undistortPoints(
                    point, camera_matrix, dist_coeffs
                )[0, 0]

                x_check = normalized[0] * z
                y_check = normalized[1] * z

                print(
                    f"solvePnP: X={tvec[0, 0]:.3f}, Y={tvec[1, 0]:.3f}, Z={z:.3f} | "
                    f"Pixel-based: X={x_check:.3f}, Y={y_check:.3f}"
                )
                if not success:
                    continue

                count += 1
                x, y, z = tvec.flatten()
                roll, pitch, yaw = rotation_to_euler(rvec)

                cv2.drawFrameAxes(
                    frame, camera_matrix, dist_coeffs,
                    rvec, tvec, MARKER_SIZE / 2, 2
                )

                px, py = marker_corners[0][0].astype(int)
                px = max(0, int(px))
                py = max(70, int(py))

                lines = [
                    f"{name} ID: {marker_id}",
                    f"X:{x:.3f} Y:{y:.3f} Z:{z:.3f} m",
                    f"R:{roll:.1f} P:{pitch:.1f} Y:{yaw:.1f} deg"
                ]

                for i, line in enumerate(lines):
                    cv2.putText(
                        frame, line,
                        (px, py - 55 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2
                    )

        cv2.putText(
            frame,
            f"Detected: {count}  Rejected candidates: {rejected_count}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        cv2.imshow("OAK ArUco Pose Estimation", frame)
        cv2.imshow("Grayscale", gray)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
