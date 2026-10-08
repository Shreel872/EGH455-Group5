from pathlib import Path
import math

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# EDIT THESE TWO
# ============================================================
#This is currently running for videos, so just change to stream
VIDEO = Path(r"EGH455 Gauge videos\VID_20230814_155207.mp4")

# Model
MODEL = Path(
    r"runs\pose\runs\pose\New_GaugePose_yolov8n_3videos\weights\best.pt"
)

# ============================================================

CONF = 0.35
KEYPOINT_CONF = 0.25
IMGSZ = 640

OUTPUT = VIDEO.with_name(VIDEO.stem + "_prediction.mp4")


def angle(cx, cy, x, y):
    return math.degrees(
        math.atan2(y - cy, x - cx)
    ) % 360


def put_text(img, text, pos, scale=0.7):
    # White outline
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )

    # Black text
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


if not VIDEO.exists():
    raise FileNotFoundError(VIDEO)

if not MODEL.exists():
    raise FileNotFoundError(MODEL)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))
print("Model:", MODEL)
print("Video:", VIDEO)

model = YOLO(str(MODEL))

cap = cv2.VideoCapture(str(VIDEO))

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30

writer = cv2.VideoWriter(
    str(OUTPUT),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

cv2.namedWindow("Gauge", cv2.WINDOW_NORMAL)

while True:
    ok, frame = cap.read()

    if not ok:
        break

    result = model(
        frame,
        imgsz=IMGSZ,
        conf=CONF,
        device=0,
        verbose=False,
    )[0]

    if (
        result.boxes is not None
        and len(result.boxes) > 0
        and result.keypoints is not None
        and len(result.keypoints.xy) > 0
    ):

        # Use highest-confidence detected gauge
        det = result.boxes.conf.argmax().item()

        pts = result.keypoints.xy[det].cpu().numpy()

        if result.keypoints.conf is not None:
            kconf = result.keypoints.conf[det].cpu().numpy()
        else:
            kconf = [1, 1, 1, 1]

        # ----------------------------------------------------
        # Keypoint order from training:
        #
        # 0 = needle tip
        # 1 = needle centre
        # 2 = zero line
        # 3 = ten line
        # ----------------------------------------------------

        if len(pts) >= 4 and min(kconf) >= KEYPOINT_CONF:

            tip = pts[0]
            centre = pts[1]
            zero = pts[2]
            ten = pts[3]

            tx, ty = map(int, tip)
            cx, cy = map(int, centre)
            zx, zy = map(int, zero)
            xx, xy = map(int, ten)

            # ==================================================
            # ANGLES
            # ==================================================

            zero_angle = angle(cx, cy, zx, zy)
            ten_angle = angle(cx, cy, xx, xy)
            needle_angle = angle(cx, cy, tx, ty)

            # Total angular range representing 0 -> 10 bar
            total_sweep = (
                ten_angle - zero_angle
            ) % 360

            # Needle angle measured from the zero reference
            needle_sweep = (
                needle_angle - zero_angle
            ) % 360

            if total_sweep > 1:

                degrees_per_bar = total_sweep / 10.0

                # Normal case: needle is inside calibrated sweep
                if needle_sweep <= total_sweep:
                    pressure_bar = (
                        needle_sweep / degrees_per_bar
                    )

                # If prediction moves just outside the calibrated
                # sweep, choose closest end rather than wrapping
                # around by ~360 degrees.
                else:
                    dist_zero = min(
                        abs(needle_angle - zero_angle),
                        360 - abs(needle_angle - zero_angle),
                    )

                    dist_ten = min(
                        abs(needle_angle - ten_angle),
                        360 - abs(needle_angle - ten_angle),
                    )

                    pressure_bar = (
                        0.0
                        if dist_zero < dist_ten
                        else 10.0
                    )

                pressure_mpa = pressure_bar * 0.1

                # ==================================================
                # DRAW LINES
                # ==================================================

                # Needle - RED
                cv2.line(
                    frame,
                    (cx, cy),
                    (tx, ty),
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

                # Zero - BLUE
                cv2.line(
                    frame,
                    (cx, cy),
                    (zx, zy),
                    (255, 0, 0),
                    3,
                    cv2.LINE_AA,
                )

                # Ten - YELLOW
                cv2.line(
                    frame,
                    (cx, cy),
                    (xx, xy),
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

                # ==================================================
                # DRAW KEYPOINTS
                # ==================================================

                cv2.circle(
                    frame,
                    (tx, ty),
                    7,
                    (0, 0, 255),
                    -1,
                )

                cv2.circle(
                    frame,
                    (cx, cy),
                    7,
                    (0, 255, 0),
                    -1,
                )

                cv2.circle(
                    frame,
                    (zx, zy),
                    7,
                    (255, 0, 0),
                    -1,
                )

                cv2.circle(
                    frame,
                    (xx, xy),
                    7,
                    (0, 255, 255),
                    -1,
                )

                # Labels
                put_text(
                    frame,
                    "TIP",
                    (tx + 10, ty),
                    0.6,
                )

                put_text(
                    frame,
                    "CENTRE",
                    (cx + 10, cy),
                    0.6,
                )

                put_text(
                    frame,
                    "0",
                    (zx + 10, zy),
                    0.6,
                )

                put_text(
                    frame,
                    "10",
                    (xx + 10, xy),
                    0.6,
                )

                # ==================================================
                # RESULTS
                # ==================================================

                put_text(
                    frame,
                    f"Pressure: {pressure_bar:.2f} bar",
                    (30, 45),
                    0.9,
                )

                put_text(
                    frame,
                    f"Pressure: {pressure_mpa:.3f} MPa",
                    (30, 85),
                    0.9,
                )

                put_text(
                    frame,
                    f"0 -> 10: {total_sweep:.1f} deg",
                    (30, 130),
                    0.65,
                )

                put_text(
                    frame,
                    f"0 -> needle: {needle_sweep:.1f} deg",
                    (30, 160),
                    0.65,
                )

                put_text(
                    frame,
                    f"Degrees/bar: {degrees_per_bar:.2f}",
                    (30, 190),
                    0.65,
                )

    writer.write(frame)

    # Display only - does not affect saved resolution
    display = frame

    if width > 1280:
        scale = 1280 / width

        display = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
        )

    cv2.imshow("Gauge", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


cap.release()
writer.release()
cv2.destroyAllWindows()

print("\nDone")
print("Saved:", OUTPUT)