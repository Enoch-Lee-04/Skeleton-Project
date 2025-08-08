import argparse
import csv
import json
import math
import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

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


def draw_smoothed_skeleton(
    image_bgr: np.ndarray,
    points_px: List[Optional[Tuple[int, int]]],
    valid_mask: List[bool],
    connections: Sequence[Tuple[int, int]],
) -> None:
    # Draw connections first
    for a_idx, b_idx in connections:
        if a_idx < 0 or b_idx < 0 or a_idx >= len(points_px) or b_idx >= len(points_px):
            continue
        if not (valid_mask[a_idx] and valid_mask[b_idx]):
            continue
        ax, ay = points_px[a_idx]  # type: ignore[index]
        bx, by = points_px[b_idx]  # type: ignore[index]
        cv2.line(image_bgr, (ax, ay), (bx, by), (0, 255, 255), 2)

    # Draw keypoints
    for idx, (pt, is_valid) in enumerate(zip(points_px, valid_mask)):
        if not is_valid or pt is None:
            continue
        x, y = pt
        cv2.circle(image_bgr, (x, y), 3, (0, 128, 255), thickness=-1, lineType=cv2.LINE_AA)


def ema_update(
    prev_xy: Optional[Tuple[float, float]],
    curr_xy: Tuple[float, float],
    alpha: float,
) -> Tuple[float, float]:
    if prev_xy is None:
        return curr_xy
    px, py = prev_xy
    cx, cy = curr_xy
    sx = alpha * cx + (1.0 - alpha) * px
    sy = alpha * cy + (1.0 - alpha) * py
    return sx, sy


def compute_angle_degrees(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> Optional[float]:
    # Angle at A formed by points B and C
    abx, aby = b[0] - a[0], b[1] - a[1]
    acx, acy = c[0] - a[0], c[1] - a[1]
    ab_norm = math.hypot(abx, aby)
    ac_norm = math.hypot(acx, acy)
    if ab_norm == 0.0 or ac_norm == 0.0:
        return None
    dot = abx * acx + aby * acy
    cos_theta = dot / (ab_norm * ac_norm)
    # Clamp for numerical stability
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.degrees(math.acos(cos_theta))
    return theta


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


def run(
    camera_index: int,
    frame_width: int,
    frame_height: int,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    mirror: bool,
    alpha: float,
    conf_threshold: float,
    export_csv: Optional[str],
    export_json: Optional[str],
) -> None:
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

    # Smoothing state: normalized coords in [0, 1]
    num_landmarks = 33
    smoothed_norm_xy: List[Optional[Tuple[float, float]]] = [None] * num_landmarks
    valid_mask: List[bool] = [False] * num_landmarks

    # Analytics collection for export
    analytics_frames: List[Dict] = []
    start_monotonic = time.perf_counter()

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

            # Measurement update (normalized)
            measurement_norm_xy: List[Optional[Tuple[float, float]]] = [None] * num_landmarks
            measurement_conf: List[float] = [0.0] * num_landmarks
            if results.pose_landmarks:
                for idx, lm in enumerate(results.pose_landmarks.landmark[:num_landmarks]):
                    mx, my = float(lm.x), float(lm.y)
                    conf = float(getattr(lm, "visibility", 1.0) or 0.0)
                    measurement_norm_xy[idx] = (mx, my)
                    measurement_conf[idx] = conf

            # Update smoothing with confidence filtering
            for i in range(num_landmarks):
                meas = measurement_norm_xy[i]
                conf = measurement_conf[i]
                if meas is not None and conf >= conf_threshold:
                    smoothed_norm_xy[i] = ema_update(smoothed_norm_xy[i], meas, alpha)
                    valid_mask[i] = True
                else:
                    # Keep previous value (if exists); otherwise mark invalid
                    valid_mask[i] = smoothed_norm_xy[i] is not None

            # Prepare pixel coordinates for drawing
            h, w = frame_bgr.shape[:2]
            points_px: List[Optional[Tuple[int, int]]] = [None] * num_landmarks
            for i, sxy in enumerate(smoothed_norm_xy):
                if sxy is None:
                    continue
                x_px = int(round(sxy[0] * w))
                y_px = int(round(sxy[1] * h))
                points_px[i] = (x_px, y_px)

            # Draw smoothed skeleton
            mp_pose = mp.solutions.pose
            draw_smoothed_skeleton(frame_bgr, points_px, valid_mask, mp_pose.POSE_CONNECTIONS)

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

            # Compute angles (using smoothed normalized coords)
            def get_xy(idx: int) -> Optional[Tuple[float, float]]:
                return smoothed_norm_xy[idx]

            angles: Dict[str, Optional[float]] = {}
            pl = mp.solutions.pose.PoseLandmark
            # Elbows: angle at elbow with shoulder and wrist
            pairs = {
                "left_elbow": (pl.LEFT_ELBOW, pl.LEFT_SHOULDER, pl.LEFT_WRIST),
                "right_elbow": (pl.RIGHT_ELBOW, pl.RIGHT_SHOULDER, pl.RIGHT_WRIST),
                "left_knee": (pl.LEFT_KNEE, pl.LEFT_HIP, pl.LEFT_ANKLE),
                "right_knee": (pl.RIGHT_KNEE, pl.RIGHT_HIP, pl.RIGHT_ANKLE),
                "left_hip": (pl.LEFT_HIP, pl.LEFT_SHOULDER, pl.LEFT_KNEE),
                "right_hip": (pl.RIGHT_HIP, pl.RIGHT_SHOULDER, pl.RIGHT_KNEE),
            }
            for name, (a_i, b_i, c_i) in pairs.items():
                a_xy = get_xy(a_i.value)
                b_xy = get_xy(b_i.value)
                c_xy = get_xy(c_i.value)
                if a_xy is None or b_xy is None or c_xy is None:
                    angles[name] = None
                else:
                    ang = compute_angle_degrees(a_xy, b_xy, c_xy)
                    angles[name] = ang

            # Collect analytics for export
            timestamp_ms = int((frame_end_time - start_monotonic) * 1000)
            keypoints_list = []
            for i in range(num_landmarks):
                sxy = smoothed_norm_xy[i]
                conf = measurement_conf[i]
                if sxy is None:
                    keypoints_list.append({"x": None, "y": None, "confidence": conf})
                else:
                    keypoints_list.append({"x": sxy[0], "y": sxy[1], "confidence": conf})

            analytics_frames.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "keypoints": keypoints_list,
                    "angles": angles,
                }
            )

            cv2.imshow("Baseline Pose (Press 'q' to quit)", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

        # Export analytics if requested
        if export_json:
            try:
                with open(export_json, "w", encoding="utf-8") as f:
                    json.dump(analytics_frames, f)
            except Exception as e:
                print(f"Failed to write JSON export to {export_json}: {e}")
        if export_csv:
            try:
                angle_keys = [
                    "left_elbow",
                    "right_elbow",
                    "left_knee",
                    "right_knee",
                    "left_hip",
                    "right_hip",
                ]
                # Build header
                headers = ["timestamp_ms"]
                for i in range(num_landmarks):
                    headers.extend([f"kp{i}_x", f"kp{i}_y", f"kp{i}_conf"])
                headers.extend(angle_keys)
                with open(export_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    for frame in analytics_frames:
                        row = [frame["timestamp_ms"]]
                        for i in range(num_landmarks):
                            kp = frame["keypoints"][i]
                            row.extend([
                                kp.get("x"),
                                kp.get("y"),
                                kp.get("confidence"),
                            ])
                        for ak in angle_keys:
                            row.append(frame["angles"].get(ak))
                        writer.writerow(row)
            except Exception as e:
                print(f"Failed to write CSV export to {export_csv}: {e}")


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
    parser.add_argument("--alpha", type=float, default=0.3, help="EMA smoothing factor alpha (0-1)")
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.3,
        help="Ignore measurements below this confidence for smoothing",
    )
    parser.add_argument("--export_json", type=str, default=None, help="Path to write analytics JSON")
    parser.add_argument("--export_csv", type=str, default=None, help="Path to write analytics CSV")
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
        alpha=args.alpha,
        conf_threshold=args.conf_threshold,
        export_csv=args.export_csv,
        export_json=args.export_json,
    )


