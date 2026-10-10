
import cv2
import depthai as dai
import numpy as np

# Configuration
CHECKERBOARD = (9, 6)  # Internal corners (columns, rows)
SQUARE_SIZE = 0.017     # 17 mm in metres
WIDTH, HEIGHT = 1280, 720
FPS = 30

# Checkerboard 3D coordinates
object_points = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    dtype=np.float32
)
object_points[:, :2] = (
    np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]]
    .T.reshape(-1, 2) * SQUARE_SIZE
)

objpoints = []
imgpoints = []

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

    pipeline.start()

    print("SPACE = Capture checkerboard")
    print("C     = Calibrate and save")
    print("Q     = Quit")
    print("Capture 20-30 images at different positions and angles.")

    while pipeline.isRunning():

        frame = queue.get().getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )

        found, corners = cv2.findChessboardCornersSB(
            gray,
            CHECKERBOARD,
            flags=flags
        )

        # Fallback to the traditional OpenCV detector
        if not found:
            found, corners = cv2.findChessboardCorners(
                gray,
                CHECKERBOARD,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                    | cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if found:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001
                    )
                )

        display = frame.copy()

        if found:
            cv2.drawChessboardCorners(
                display, CHECKERBOARD, corners, found
            )

        status = "FOUND" if found else "NOT FOUND"

        cv2.putText(
            display,
            f"Checkerboard: {status} | Samples: {len(objpoints)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0) if found else (0, 0, 255), 2
        )

        cv2.putText(
            display,
            "SPACE: Capture | C: Calibrate | Q: Quit",
            (20, HEIGHT - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2
        )

        cv2.imshow("OAK Camera Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and found:
            objpoints.append(object_points.copy())
            imgpoints.append(corners.copy())
            print(f"Captured sample {len(objpoints)}")

        elif key == ord("c"):

            if len(objpoints) < 15:
                print("Capture at least 15 samples first.")
                continue

            print("Calibrating camera...")

            rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
                cv2.calibrateCamera(
                    objpoints,
                    imgpoints,
                    (WIDTH, HEIGHT),
                    None,
                    None
                )
            )

            print("\nCalibration complete")
            print("RMS reprojection error:", rms)
            print("\nCamera matrix:\n", camera_matrix)
            print("\nDistortion coefficients:\n", dist_coeffs)

            np.savez(
                "camera_calibration.npz",
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                width=WIDTH,
                height=HEIGHT,
                rms=rms
            )

            print("\nSaved to camera_calibration.npz")

        elif key == ord("q"):
            break

cv2.destroyAllWindows()
