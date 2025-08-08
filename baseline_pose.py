import argparse
import time
from collections import deque

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit(
        "mediapipe is not installed. Install dependencies first: pip install -r requirements.txt"
    ) from exc


def create_pose_estimator(model_complexity: int,
                        min_detection_confidence: float,
                        min_tracking_confidence: float):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return pose


def draw_landmarks(image_bgr: np.ndarray, results) -> None:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    if results.pose_landmarks:
        # Some MediaPipe versions don't expose get_default_pose_connections_style.
        # Use explicit DrawingSpec for connections instead.
        connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
        mp_drawing.draw_landmarks(
            image=image_bgr,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=connection_spec,
        )


def put_overlay_text(image_bgr: np.ndarray, text: str, origin=(10, 30)) -> None:
    cv2.putText(
        image_bgr,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def run(camera_index: int,
        frame_width: int,
        frame_height: int,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        mirror: bool) -> None:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if frame_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    if frame_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}. Try a different --camera value.")

    pose = create_pose_estimator(
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS smoothing over last N frames
    frame_times = deque(maxlen=30)

    try:
        while True:
            frame_start_time = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Inference
            infer_start = time.perf_counter()
            results = pose.process(frame_rgb)
            infer_end = time.perf_counter()
            infer_ms = (infer_end - infer_start) * 1000.0

            # Draw landmarks and connections
            draw_landmarks(frame_bgr, results)

            # FPS calculation
            frame_end_time = time.perf_counter()
            frame_dt = frame_end_time - frame_start_time
            frame_times.append(frame_dt)
            if len(frame_times) > 0:
                avg_dt = sum(frame_times) / len(frame_times)
                fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
            else:
                fps = 0.0

            put_overlay_text(frame_bgr, f"FPS: {fps:5.1f} | Inference: {infer_ms:6.1f} ms")

            cv2.imshow("Baseline Pose (Press 'q' to quit)", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline real-time pose estimation with MediaPipe.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height (default: 720)")
    parser.add_argument(
        "--model_complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe model complexity: 0=Lite, 1=Full, 2=Heavy (default: 1)",
    )
    parser.add_argument("--min_detection", type=float, default=0.5, help="Minimum detection confidence")
    parser.add_argument("--min_tracking", type=float, default=0.5, help="Minimum tracking confidence")
    parser.add_argument("--mirror", action="store_true", help="Mirror the preview (selfie view)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
        mirror=args.mirror,
    )


