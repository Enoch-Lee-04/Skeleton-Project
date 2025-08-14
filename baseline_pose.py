import argparse
import csv
import json
import math
import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import cv2 as _cv2
from typing import Any
import numpy as np
# Treat cv2 as dynamic to silence missing-attribute warnings in static analysis
cv2: Any = _cv2

# OpenCV utility helpers to avoid static linter attribute errors
def _cv2_call(name: str, *fn_args: Any, **kwargs: Any) -> Any:
    func = getattr(cv2, name, None)
    if func is None:
        raise AttributeError(f"cv2 has no attribute {name}")
    return func(*fn_args, **kwargs)

CV_LINE_AA = getattr(cv2, "LINE_AA", 16)
CV_FONT_SIMPLEX = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)

def _concat_h(a: Any, b: Any) -> Any:
    hconcat = getattr(cv2, "hconcat", None)
    if hconcat is not None:
        return hconcat([a, b])
    return np.hstack([a, b])

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


def create_holistic_estimator(model_complexity: int,
                            min_detection_confidence: float,
                            min_tracking_confidence: float,
                            refine_face_landmarks: bool):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        refine_face_landmarks=refine_face_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return holistic


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
        _cv2_call("line", image_bgr, (ax, ay), (bx, by), (0, 255, 255), 2)

    # Draw keypoints
    for pt, is_valid in zip(points_px, valid_mask):
        if not is_valid or pt is None:
            continue
        x, y = pt
        _cv2_call("circle", image_bgr, (x, y), 3, (0, 128, 255), thickness=-1, lineType=CV_LINE_AA)


def draw_hand_dense_vectors(
    image_bgr: np.ndarray,
    points_px: List[Optional[Tuple[int, int]]],
    valid_mask: List[bool],
) -> None:
    # Additional hand overlays for dense mode: finger direction arrows and palm/fingertip polygons
    if not points_px or len(points_px) < 21:
        return
    def is_valid(i: int) -> bool:
        return 0 <= i < len(points_px) and valid_mask[i] and points_px[i] is not None
    # Draw arrows from MCP to TIP for each finger
    mcp_to_tip = [(2, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
    for mcp, tip in mcp_to_tip:
        if is_valid(mcp) and is_valid(tip):
            ax, ay = points_px[mcp]  # type: ignore[index]
            bx, by = points_px[tip]  # type: ignore[index]
            _cv2_call(
                "arrowedLine",
                image_bgr,
                (ax, ay),
                (bx, by),
                (255, 0, 255),
                2,
                CV_LINE_AA,
                0,
                0.25,
            )
    # Palm polygon across MCP joints
    palm = [5, 9, 13, 17]
    palm_cycle = palm + palm[:1]
    for i in range(len(palm_cycle) - 1):
        a = palm_cycle[i]
        b = palm_cycle[i + 1]
        if is_valid(a) and is_valid(b):
            ax, ay = points_px[a]  # type: ignore[index]
            bx, by = points_px[b]  # type: ignore[index]
            _cv2_call("line", image_bgr, (ax, ay), (bx, by), (255, 0, 255), 1)
    # Fingertip outline polygon
    tips = [8, 12, 16, 20]
    tips_cycle = tips + tips[:1]
    for i in range(len(tips_cycle) - 1):
        a = tips_cycle[i]
        b = tips_cycle[i + 1]
        if is_valid(a) and is_valid(b):
            ax, ay = points_px[a]  # type: ignore[index]
            bx, by = points_px[b]  # type: ignore[index]
            _cv2_call("line", image_bgr, (ax, ay), (bx, by), (200, 0, 200), 1)


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
    _cv2_call(
        "putText",
        image_bgr,
        text,
        origin,
        CV_FONT_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        CV_LINE_AA,
    )


def get_pose_connections(dense: bool) -> List[Tuple[int, int]]:
    pl = mp.solutions.pose.PoseLandmark
    # Base MediaPipe connections (may be enums)
    base_pairs = list(mp.solutions.pose.POSE_CONNECTIONS)

    def to_idx_pair(p: Tuple[Any, Any]) -> Tuple[int, int]:
        a, b = p
        ai = a.value if hasattr(a, "value") else int(a)
        bi = b.value if hasattr(b, "value") else int(b)
        return ai, bi

    pairs_idx: List[Tuple[int, int]] = [to_idx_pair(p) for p in base_pairs]
    if not dense:
        return pairs_idx

    extra: List[Tuple[int, int]] = []
    # Symmetry bars across left/right for better structure
    lr_pairs = [
        (pl.LEFT_EYE, pl.RIGHT_EYE),
        (pl.LEFT_EAR, pl.RIGHT_EAR),
        (pl.LEFT_SHOULDER, pl.RIGHT_SHOULDER),
        (pl.LEFT_ELBOW, pl.RIGHT_ELBOW),
        (pl.LEFT_WRIST, pl.RIGHT_WRIST),
        (pl.LEFT_HIP, pl.RIGHT_HIP),
        (pl.LEFT_KNEE, pl.RIGHT_KNEE),
        (pl.LEFT_ANKLE, pl.RIGHT_ANKLE),
        (pl.LEFT_HEEL, pl.RIGHT_HEEL),
        (pl.LEFT_FOOT_INDEX, pl.RIGHT_FOOT_INDEX),
    ]
    # Torso diagonals
    torso_diagonals = [
        (pl.LEFT_SHOULDER, pl.RIGHT_HIP),
        (pl.RIGHT_SHOULDER, pl.LEFT_HIP),
    ]
    # Head/torso anchors improve stability perception
    head_links = [
        (pl.NOSE, pl.LEFT_SHOULDER),
        (pl.NOSE, pl.RIGHT_SHOULDER),
        (pl.NOSE, pl.LEFT_HIP),
        (pl.NOSE, pl.RIGHT_HIP),
    ]
    # Leg cross-bars
    leg_cross = [
        (pl.LEFT_KNEE, pl.RIGHT_ANKLE),
        (pl.RIGHT_KNEE, pl.LEFT_ANKLE),
    ]
    for group in (lr_pairs, torso_diagonals, head_links, leg_cross):
        for a, b in group:
            extra.append((a.value, b.value))

    # Merge unique
    seen = set(pairs_idx)
    for p in extra:
        if p not in seen and (p[1], p[0]) not in seen:
            pairs_idx.append(p)
            seen.add(p)
    return pairs_idx


class SquatFSM:
    """Finite-state machine for squat repetition detection using knee angle and hip/knee vertical delta."""

    def __init__(
        self,
        bottom_knee_angle_threshold: float = 80.0,
        top_knee_angle_threshold: float = 160.0,
        hip_knee_min_delta_px: int = 20,
        tempo_min_seconds: float = 0.5,
    ) -> None:
        self.bottom_threshold = float(bottom_knee_angle_threshold)
        self.top_threshold = float(top_knee_angle_threshold)
        self.hip_knee_min_delta_px = int(hip_knee_min_delta_px)
        self.tempo_min_seconds = float(tempo_min_seconds)

        self.state: str = "TOP"
        self.rep_count: int = 0
        self.visited_bottom_in_rep: bool = False
        self.rep_start_time: Optional[float] = None
        self.last_bottom_time: Optional[float] = None
        self.min_knee_angle_in_rep: Optional[float] = None

        self.reps_durations: List[float] = []
        self.reps_depth_angles: List[float] = []
        self.too_fast_count: int = 0
        self.last_feedback: str = ""

    def _depth_reached(self, knee_angle: float, hip_y_px: int, knee_y_px: int) -> bool:
        vertical_delta = hip_y_px - knee_y_px  # +ve when hip is lower (in image coords)
        return (
            knee_angle <= self.bottom_threshold and vertical_delta >= self.hip_knee_min_delta_px
        )

    def update(
        self,
        now_seconds: float,
        knee_angle: Optional[float],
        hip_y_px: Optional[int],
        knee_y_px: Optional[int],
    ) -> None:
        # Missing data: keep state but don't transition
        if knee_angle is None or hip_y_px is None or knee_y_px is None:
            self.last_feedback = "No data"
            return

        at_top = knee_angle >= self.top_threshold
        depth_reached = self._depth_reached(knee_angle, hip_y_px, knee_y_px)

        # Track min knee angle within a rep window
        if self.rep_start_time is not None:
            if self.min_knee_angle_in_rep is None:
                self.min_knee_angle_in_rep = knee_angle
            else:
                if knee_angle < self.min_knee_angle_in_rep:
                    self.min_knee_angle_in_rep = knee_angle

        if self.state == "TOP":
            if not at_top:
                self.state = "GOING_DOWN"
                self.rep_start_time = now_seconds
                self.visited_bottom_in_rep = False
                self.min_knee_angle_in_rep = knee_angle
            return

        if self.state == "GOING_DOWN":
            if depth_reached:
                self.state = "BOTTOM"
                self.last_bottom_time = now_seconds
                self.visited_bottom_in_rep = True
            return

        if self.state == "BOTTOM":
            if not depth_reached:
                self.state = "GOING_UP"
            return

        if self.state == "GOING_UP":
            if at_top:
                # Only count if full cycle with bottom visited
                if self.visited_bottom_in_rep and self.rep_start_time is not None:
                    rep_time = now_seconds - self.rep_start_time
                    self.reps_durations.append(rep_time)
                    depth_angle = (
                        self.min_knee_angle_in_rep if self.min_knee_angle_in_rep is not None else knee_angle
                    )
                    self.reps_depth_angles.append(depth_angle)
                    self.rep_count += 1

                    if self.last_bottom_time is not None and (now_seconds - self.last_bottom_time) < self.tempo_min_seconds:
                        self.too_fast_count += 1
                        self.last_feedback = "Too fast"
                    elif depth_angle > self.bottom_threshold:
                        self.last_feedback = "Too shallow"
                    else:
                        self.last_feedback = "OK"
                else:
                    self.last_feedback = "Incomplete"

                # Reset for next rep
                self.state = "TOP"
                self.rep_start_time = None
                self.visited_bottom_in_rep = False
                self.min_knee_angle_in_rep = None
            return

    def summary_metrics(self) -> Dict[str, Optional[float]]:
        def safe_mean(values: List[float]) -> Optional[float]:
            return float(sum(values) / len(values)) if values else None

        def safe_std(values: List[float]) -> Optional[float]:
            if len(values) < 2:
                return None
            mean_v = sum(values) / len(values)
            var = sum((v - mean_v) ** 2 for v in values) / (len(values) - 1)
            return float(math.sqrt(var))

        return {
            "reps": float(self.rep_count),
            "avg_depth_angle": safe_mean(self.reps_depth_angles),
            "avg_rep_time_s": safe_mean(self.reps_durations),
            "std_rep_time_s": safe_std(self.reps_durations),
            "too_fast_count": float(self.too_fast_count),
        }


class ActionRecognizer:
    def __init__(self, window_seconds: float = 1.5) -> None:
        self.window_seconds = float(window_seconds)
        self.history: deque = deque()
        self.current_label: str = "Unknown"
        self.current_confidence: float = 0.0

    def update(
        self,
        now_seconds: float,
        smoothed_norm_xy: List[Optional[Tuple[float, float]]],
        points_px: List[Optional[Tuple[int, int]]],
    ) -> None:
        self.history.append({
            "t": now_seconds,
            "norm": smoothed_norm_xy.copy(),
            "px": points_px.copy(),
        })
        t_min = now_seconds - self.window_seconds
        while self.history and self.history[0]["t"] < t_min:
            self.history.popleft()
        self._classify()

    def _get_series(self, idx: int, coord: int = 0) -> List[float]:
        vals: List[float] = []
        for f in self.history:
            arr = f["norm"]
            v = arr[idx]
            if v is None:
                vals.append(float("nan"))
            else:
                vals.append(float(v[coord]))
        return vals

    def _shoulder_width_px(self) -> Optional[float]:
        if not self.history:
            return None
        frame = self.history[-1]
        px_list = frame["px"]
        try:
            pl = mp.solutions.pose.PoseLandmark
            ls = px_list[pl.LEFT_SHOULDER.value]
            rs = px_list[pl.RIGHT_SHOULDER.value]
            if ls is None or rs is None:
                return None
            return float(math.hypot(ls[0] - rs[0], ls[1] - rs[1]))
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    @staticmethod
    def _nan_safe_range(series: List[float]) -> Optional[float]:
        clean = [v for v in series if not math.isnan(v)]
        if not clean:
            return None
        return float(max(clean) - min(clean))

    @staticmethod
    def _nan_safe_mean(series: List[float]) -> Optional[float]:
        clean = [v for v in series if not math.isnan(v)]
        if not clean:
            return None
        return float(sum(clean) / len(clean))

    @staticmethod
    def _count_sign_changes(series: List[float], deadband: float = 0.005) -> int:
        prev_sign = 0
        changes = 0
        for v in series:
            if math.isnan(v):
                continue
            s = 1 if v > deadband else (-1 if v < -deadband else 0)
            if s != 0 and prev_sign != 0 and s != prev_sign:
                changes += 1
            if s != 0:
                prev_sign = s
        return changes

    def _classify(self) -> None:
        self.current_label = "Unknown"
        self.current_confidence = 0.0
        if len(self.history) < 3:
            return

        pl = mp.solutions.pose.PoseLandmark
        # Waving detector
        lwx = self._get_series(pl.LEFT_WRIST.value, 0)
        rwx = self._get_series(pl.RIGHT_WRIST.value, 0)
        lwy = self._get_series(pl.LEFT_WRIST.value, 1)
        rwy = self._get_series(pl.RIGHT_WRIST.value, 1)
        lsy = self._get_series(pl.LEFT_SHOULDER.value, 1)
        rsy = self._get_series(pl.RIGHT_SHOULDER.value, 1)
        shoulder_y_mean_left = self._nan_safe_mean(lsy)
        shoulder_y_mean_right = self._nan_safe_mean(rsy)
        shoulder_y_mean = None
        if shoulder_y_mean_left is not None and shoulder_y_mean_right is not None:
            shoulder_y_mean = (shoulder_y_mean_left + shoulder_y_mean_right) / 2.0

        sw_px = self._shoulder_width_px() or 1.0
        wave_score = 0.0
        if shoulder_y_mean is not None:
            def above_shoulder(ys: List[float]) -> float:
                clean = [y for y in ys if not math.isnan(y)]
                if not clean:
                    return 0.0
                count = sum(1 for y in clean if y < shoulder_y_mean - 0.03)
                return count / len(clean)

            left_above = above_shoulder(lwy)
            right_above = above_shoulder(rwy)
            left_range = self._nan_safe_range(lwx) or 0.0
            right_range = self._nan_safe_range(rwx) or 0.0
            # Normalize horizontal range roughly by image scale using shoulder width proxy
            # Convert normalized x-range to px using current width estimate if available
            # Here we approximate as already normalized; scale factor not strictly needed
            horiz_motion = max(left_range, right_range)
            above_prop = max(left_above, right_above)
            if above_prop > 0.5 and horiz_motion > 0.08:
                wave_score = min(1.0, 0.5 * above_prop + 3.0 * (horiz_motion - 0.08))

        # Pointing detector (arm extended and stable)
        # Use elbow angle and wrist distance from shoulder
        # elbow indices used via PoseLandmark directly in angle_now()
        shoulder_left = pl.LEFT_SHOULDER.value
        shoulder_right = pl.RIGHT_SHOULDER.value
        wrist_left = pl.LEFT_WRIST.value
        wrist_right = pl.RIGHT_WRIST.value

        def angle_now(name: str) -> Optional[float]:
            frame = self.history[-1]
            # Best-effort compute current elbow angles from norm coords
            def get_xy(i: int) -> Optional[Tuple[float, float]]:
                v = frame["norm"][i]
                return None if v is None else (float(v[0]), float(v[1]))
            if name == "left_elbow":
                a = get_xy(pl.LEFT_ELBOW.value)
                b = get_xy(pl.LEFT_SHOULDER.value)
                c = get_xy(pl.LEFT_WRIST.value)
            else:
                a = get_xy(pl.RIGHT_ELBOW.value)
                b = get_xy(pl.RIGHT_SHOULDER.value)
                c = get_xy(pl.RIGHT_WRIST.value)
            if a is None or b is None or c is None:
                return None
            return compute_angle_degrees(a, b, c)

        left_elbow_deg = angle_now("left_elbow")
        right_elbow_deg = angle_now("right_elbow")

        px_list = self.history[-1]["px"]
        def wrist_shoulder_dist_px(wrist_idx: int, shoulder_idx: int) -> Optional[float]:
            w = px_list[wrist_idx]
            s = px_list[shoulder_idx]
            if w is None or s is None:
                return None
            return float(math.hypot(w[0] - s[0], w[1] - s[1]))

        dist_l = wrist_shoulder_dist_px(wrist_left, shoulder_left)
        dist_r = wrist_shoulder_dist_px(wrist_right, shoulder_right)
        stability_l = self._nan_safe_range(lwx) or 0.0
        stability_r = self._nan_safe_range(rwx) or 0.0
        point_score = 0.0
        if sw_px and (left_elbow_deg is not None or right_elbow_deg is not None):
            cond_l = (left_elbow_deg or 0.0) > 160.0 and (dist_l or 0.0) > 0.8 * sw_px and stability_l < 0.04
            cond_r = (right_elbow_deg or 0.0) > 160.0 and (dist_r or 0.0) > 0.8 * sw_px and stability_r < 0.04
            if cond_l or cond_r:
                point_score = 0.6 + 0.4 * (max((dist_l or 0.0), (dist_r or 0.0)) / max(sw_px, 1.0))
                point_score = min(1.0, point_score)

        # Gait detector: running vs walking via ankle y-difference sign changes
        lay = self._get_series(pl.LEFT_ANKLE.value, 1)
        ray = self._get_series(pl.RIGHT_ANKLE.value, 1)
        diff = []
        for i in range(min(len(lay), len(ray))):
            a = lay[i]
            b = ray[i]
            if math.isnan(a) or math.isnan(b):
                diff.append(float("nan"))
            else:
                diff.append(a - b)
        sign_changes = self._count_sign_changes(diff, deadband=0.007)
        duration = self.history[-1]["t"] - self.history[0]["t"]
        freq_hz = (sign_changes / 2.0) / max(duration, 1e-6)
        walk_score = 0.0
        run_score = 0.0
        if freq_hz > 0.6:
            if freq_hz > 1.8:
                run_score = min(1.0, (freq_hz - 1.8) / 0.8)
            else:
                walk_score = min(1.0, (freq_hz - 0.6) / 1.0)

        # Choose label by highest score with simple priority
        candidates = [
            ("Waving", wave_score),
            ("Pointing", point_score),
            ("Running", run_score),
            ("Walking", walk_score),
        ]
        label, score = max(candidates, key=lambda x: x[1])
        if score > 0.5:
            self.current_label = label
            self.current_confidence = float(score)
        else:
            self.current_label = "Unknown"
            self.current_confidence = float(score)


class Gesture:
    def __init__(
        self,
        name: str,
        enter_fn,
        exit_fn,
        min_frames: int = 5,
        cooldown_frames: int = 20,
    ) -> None:
        self.name = name
        self.enter_fn = enter_fn
        self.exit_fn = exit_fn
        self.min_frames = int(min_frames)
        self.cooldown_frames = int(cooldown_frames)
        self.state: str = "idle"  # idle|candidate|active|cooldown
        self.counter: int = 0
        self.cooldown: int = 0

    def step(self, kp: Dict[str, Tuple[float, float, float]], feats: Dict[str, float]) -> bool:
        if self.cooldown > 0:
            self.cooldown -= 1
            self.state = "cooldown"
            self.counter = 0
            return False
        if self.state in ("idle", "candidate"):
            if self.enter_fn(kp, feats):
                self.counter += 1
                self.state = "candidate"
                if self.counter >= self.min_frames:
                    self.state = "active"
                    self.cooldown = self.cooldown_frames
                    self.counter = 0
                    return True
            else:
                self.counter = 0
                self.state = "idle"
            return False
        if self.state == "active":
            if self.exit_fn(kp, feats):
                self.state = "cooldown"
            return False
        return False


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _build_keypoints_dict(points_px: List[Optional[Tuple[int, int]]], smoothed_norm_xy: List[Optional[Tuple[float, float]]]) -> Optional[Dict[str, Tuple[float, float, float]]]:
    try:
        pl = mp.solutions.pose.PoseLandmark
        idx_map = {
            "nose": pl.NOSE.value,
            "left_shoulder": pl.LEFT_SHOULDER.value,
            "right_shoulder": pl.RIGHT_SHOULDER.value,
            "left_elbow": pl.LEFT_ELBOW.value,
            "right_elbow": pl.RIGHT_ELBOW.value,
            "left_wrist": pl.LEFT_WRIST.value,
            "right_wrist": pl.RIGHT_WRIST.value,
            "left_hip": pl.LEFT_HIP.value,
            "right_hip": pl.RIGHT_HIP.value,
            "left_ankle": pl.LEFT_ANKLE.value,
            "right_ankle": pl.RIGHT_ANKLE.value,
        }
        result: Dict[str, Tuple[float, float, float]] = {}
        for name, idx in idx_map.items():
            px = points_px[idx]
            nm = smoothed_norm_xy[idx]
            if px is None or nm is None:
                return None
            # Visibility/confidence not given here; approximate as presence
            result[name] = (float(px[0]), float(px[1]), 1.0)

        # Mid hip synthetic
        lhip = result["left_hip"]
        rhip = result["right_hip"]
        mid_hip_xy = ((lhip[0] + rhip[0]) / 2.0, (lhip[1] + rhip[1]) / 2.0)
        result["mid_hip"] = (mid_hip_xy[0], mid_hip_xy[1], min(lhip[2], rhip[2]))
        return result
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _features_from_kp(kp: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
    left_shoulder = (kp["left_shoulder"][0], kp["left_shoulder"][1])
    right_shoulder = (kp["right_shoulder"][0], kp["right_shoulder"][1])
    mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2.0, (left_shoulder[1] + right_shoulder[1]) / 2.0)
    mid_hip = (kp["mid_hip"][0], kp["mid_hip"][1])
    torso = _distance(mid_shoulder, mid_hip)
    shoulder_w = _distance(left_shoulder, right_shoulder)
    nose_y = kp["nose"][1]
    return {
        "torso": max(torso, 1.0),
        "shoulder_w": max(shoulder_w, 1.0),
        "nose_y": float(nose_y),
    }


def _hands_up_enter(kp: Dict[str, Tuple[float, float, float]], f: Dict[str, float]) -> bool:
    lw, rw = kp["left_wrist"], kp["right_wrist"]
    le, re = kp["left_elbow"], kp["right_elbow"]
    ls, rs = kp["left_shoulder"], kp["right_shoulder"]
    nose_y = f["nose_y"]
    t = f["torso"]
    conf_ok = min(lw[2], rw[2], le[2], re[2]) > 0.35
    above_head = (lw[1] < nose_y - 0.15 * t) and (rw[1] < nose_y - 0.15 * t)
    elbows_up = (le[1] < ls[1] - 0.05 * t) and (re[1] < rs[1] - 0.05 * t)
    return conf_ok and above_head and elbows_up


def _hands_up_exit(kp: Dict[str, Tuple[float, float, float]], f: Dict[str, float]) -> bool:
    lw, rw = kp["left_wrist"], kp["right_wrist"]
    return (lw[1] < f["nose_y"] + 0.05 * f["torso"]) and (rw[1] < f["nose_y"] + 0.05 * f["torso"])


def _tpose_enter(kp: Dict[str, Tuple[float, float, float]], f: Dict[str, float]) -> bool:
    lw, rw = kp["left_wrist"], kp["right_wrist"]
    ls, rs = kp["left_shoulder"], kp["right_shoulder"]
    le, re = kp["left_elbow"], kp["right_elbow"]
    s = f["shoulder_w"]
    level = abs(lw[1] - ls[1]) < 0.25 * s and abs(rw[1] - rs[1]) < 0.25 * s
    straight = abs(le[0] - ls[0]) > 0.7 * s and abs(re[0] - rs[0]) > 0.7 * s
    wide = abs(rw[0] - lw[0]) > 2.0 * s
    conf_ok = min(lw[2], rw[2], le[2], re[2], ls[2], rs[2]) > 0.35
    return conf_ok and level and straight and wide


def _tpose_exit(kp: Dict[str, Tuple[float, float, float]], f: Dict[str, float]) -> bool:
    return (
        kp["left_wrist"][1] < kp["left_shoulder"][1] + 0.4 * f["shoulder_w"]
        and kp["right_wrist"][1] < kp["right_shoulder"][1] + 0.4 * f["shoulder_w"]
    )


def _left_arm_up_enter(kp: Dict[str, Tuple[float, float, float]], f: Dict[str, float]) -> bool:
    lw, ls = kp["left_wrist"], kp["left_shoulder"]
    rw, rs = kp["right_wrist"], kp["right_shoulder"]
    conf_ok = min(lw[2], rw[2], ls[2], rs[2]) > 0.35
    left_up = lw[1] < ls[1] - 0.1 * f["torso"]
    right_down = rw[1] > rs[1] + 0.05 * f["torso"]
    return conf_ok and left_up and right_down


def _left_arm_up_exit(kp: Dict[str, Tuple[float, float, float]], f: Dict[str, float]) -> bool:
    return kp["left_wrist"][1] < kp["left_shoulder"][1] + 0.1 * f["torso"]


def _thumbs_up_enter(
    _kp: Dict[str, Tuple[float, float, float]],
    _f: Dict[str, float],
    hand_points_l: Optional[List[Optional[Tuple[int, int]]]] = None,
    hand_points_r: Optional[List[Optional[Tuple[int, int]]]] = None,
) -> bool:
    # Simple heuristic: thumb tip above thumb MCP; other fingertips below their MCPs (folded)
    # Requires hand landmarks
    try:
        def is_thumb_up(hand_pts: Optional[List[Optional[Tuple[int, int]]]]) -> bool:
            if hand_pts is None:
                return False
            # Indices: THUMB_TIP=4, THUMB_IP=3, THUMB_MCP=2; INDEX_TIP=8, MIDDLE_TIP=12, RING_TIP=16, PINKY_TIP=20; corresponding MCP: 5,9,13,17
            req = [2, 4, 5, 8, 9, 12, 13, 16, 17, 20]
            for r in req:
                if r >= len(hand_pts) or hand_pts[r] is None:
                    return False
            thumb_tip = hand_pts[4][1]
            thumb_mcp = hand_pts[2][1]
            # y smaller is higher
            thumb_up = thumb_tip < thumb_mcp - 5
            index_folded = hand_pts[8][1] > hand_pts[5][1] + 5
            middle_folded = hand_pts[12][1] > hand_pts[9][1] + 5
            ring_folded = hand_pts[16][1] > hand_pts[13][1] + 5
            pinky_folded = hand_pts[20][1] > hand_pts[17][1] + 5
            return bool(thumb_up and index_folded and middle_folded and ring_folded and pinky_folded)

        return is_thumb_up(hand_points_l) or is_thumb_up(hand_points_r)
    except (IndexError, TypeError, ValueError):
        return False


def _thumbs_up_exit(
    _kp: Dict[str, Tuple[float, float, float]],
    _f: Dict[str, float],
    _hand_points_l: Optional[List[Optional[Tuple[int, int]]]] = None,
    _hand_points_r: Optional[List[Optional[Tuple[int, int]]]] = None,
) -> bool:
    return True  # single-shot; exit immediately after active


class GestureEngine:
    def __init__(self, min_frames: int, cooldown_frames: int, use_hands: bool) -> None:
        self.use_hands = use_hands
        self.gestures: List[Gesture] = [
            Gesture("PAUSE_PLAY", _hands_up_enter, _hands_up_exit, min_frames=min_frames, cooldown_frames=cooldown_frames),
            Gesture("NEXT_SLIDE", _tpose_enter, _tpose_exit, min_frames=min_frames, cooldown_frames=cooldown_frames),
            Gesture("MUTE_TOGGLE", _left_arm_up_enter, _left_arm_up_exit, min_frames=min_frames, cooldown_frames=cooldown_frames),
        ]
        if self.use_hands:
            # Wrap thumbs-up to pass hand pts
            def enter_thumb(kp, f):
                return _thumbs_up_enter(kp, f, self.last_lh_points, self.last_rh_points)
            def exit_thumb(kp, f):
                return _thumbs_up_exit(kp, f, self.last_lh_points, self.last_rh_points)
            self.gestures.append(Gesture("CONFIRM", enter_thumb, exit_thumb, min_frames=max(3, min_frames-1), cooldown_frames=cooldown_frames))

        self.last_lh_points: Optional[List[Optional[Tuple[int, int]]]] = None
        self.last_rh_points: Optional[List[Optional[Tuple[int, int]]]] = None

    def update_hand_points(
        self,
        lh_points_px: Optional[List[Optional[Tuple[int, int]]]],
        rh_points_px: Optional[List[Optional[Tuple[int, int]]]],
    ) -> None:
        self.last_lh_points = lh_points_px
        self.last_rh_points = rh_points_px

    def step(self, points_px: List[Optional[Tuple[int, int]]], smoothed_norm_xy: List[Optional[Tuple[float, float]]]) -> List[str]:
        kp = _build_keypoints_dict(points_px, smoothed_norm_xy)
        if kp is None:
            return []
        feats = _features_from_kp(kp)
        fired: List[str] = []
        for g in self.gestures:
            if g.step(kp, feats):
                fired.append(g.name)
        return fired

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
    movement: str,
    side_by_side: bool,
    squat_bottom_angle: float,
    squat_top_angle: float,
    hip_knee_delta_px: int,
    tempo_min_s: float,
    enable_actions: bool,
    action_window_s: float,
    enable_hands: bool,
    enable_face: bool,
    gesture_min_frames: int,
    gesture_cooldown_frames: int,
    pose_density: str,
) -> None:
    VideoCapture = getattr(cv2, "VideoCapture")
    if hasattr(cv2, "CAP_DSHOW"):
        cap = VideoCapture(camera_index, getattr(cv2, "CAP_DSHOW"))
    else:
        cap = VideoCapture(camera_index)
    if frame_width > 0:
        cap.set(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3), frame_width)
    if frame_height > 0:
        cap.set(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4), frame_height)

    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}. Try a different --camera value.")

    # Choose initial estimator mode; dense pose requires holistic for face mesh
    estimator_mode: str = "holistic" if (enable_hands or enable_face or pose_density == "dense") else "pose"
    if estimator_mode == "holistic":
        estimator = create_holistic_estimator(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_face_landmarks=(enable_face or pose_density == "dense"),
        )
    else:
        estimator = create_pose_estimator(
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

    # Optional extra landmark sets
    lh_count, rh_count, face_count = 21, 21, 468
    smoothed_lh_norm_xy: List[Optional[Tuple[float, float]]] = [None] * lh_count
    smoothed_rh_norm_xy: List[Optional[Tuple[float, float]]] = [None] * rh_count
    smoothed_face_norm_xy: List[Optional[Tuple[float, float]]] = [None] * face_count
    valid_lh_mask: List[bool] = [False] * lh_count
    valid_rh_mask: List[bool] = [False] * rh_count
    valid_face_mask: List[bool] = [False] * face_count

    # Analytics collection for export
    analytics_frames: List[Dict] = []
    start_monotonic = time.perf_counter()

    # Initialize movement FSM(s)
    squat_fsm: Optional[SquatFSM] = None
    if movement.lower() == "squat":
        squat_fsm = SquatFSM(
            bottom_knee_angle_threshold=squat_bottom_angle,
            top_knee_angle_threshold=squat_top_angle,
            hip_knee_min_delta_px=hip_knee_delta_px,
            tempo_min_seconds=tempo_min_s,
        )

    # Initialize action recognizer
    action_recognizer: Optional[ActionRecognizer] = ActionRecognizer(action_window_s) if enable_actions else None
    gesture_engine: Optional[GestureEngine] = GestureEngine(gesture_min_frames, gesture_cooldown_frames, enable_hands) if (enable_actions or True) else None
    
    # Runtime toggle state
    current_pose_density = pose_density

    try:
        while True:
            frame_start_time = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if mirror:
                frame_bgr = _cv2_call("flip", frame_bgr, 1)

            # Convert to RGB for MediaPipe
            frame_rgb = _cv2_call("cvtColor", frame_bgr, getattr(cv2, "COLOR_BGR2RGB", 4))

            # Switch estimator dynamically if density/flags require holistic vs pose
            use_holistic_needed = bool(enable_hands or enable_face or current_pose_density == "dense")
            if (estimator_mode == "pose" and use_holistic_needed) or (estimator_mode == "holistic" and not use_holistic_needed):
                try:
                    estimator.close()
                except (AttributeError, RuntimeError):
                    pass
                if use_holistic_needed:
                    estimator = create_holistic_estimator(
                        model_complexity=model_complexity,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence,
                        refine_face_landmarks=(enable_face or current_pose_density == "dense"),
                    )
                    estimator_mode = "holistic"
                else:
                    estimator = create_pose_estimator(
                        model_complexity=model_complexity,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence,
                    )
                    estimator_mode = "pose"

            # Inference
            infer_start = time.perf_counter()
            results = estimator.process(frame_rgb)
            infer_end = time.perf_counter()
            infer_ms = (infer_end - infer_start) * 1000.0

            # Measurement update (normalized)
            measurement_norm_xy: List[Optional[Tuple[float, float]]] = [None] * num_landmarks
            measurement_conf: List[float] = [0.0] * num_landmarks
            if getattr(results, "pose_landmarks", None):
                for idx, lm in enumerate(results.pose_landmarks.landmark[:num_landmarks]):
                    mx, my = float(lm.x), float(lm.y)
                    conf = float(getattr(lm, "visibility", 1.0) or 0.0)
                    measurement_norm_xy[idx] = (mx, my)
                    measurement_conf[idx] = conf

            # Hands (if using holistic now)
            if use_holistic_needed and enable_hands:
                # Left hand
                if getattr(results, "left_hand_landmarks", None):
                    for idx, lm in enumerate(results.left_hand_landmarks.landmark[:lh_count]):
                        smoothed_lh_norm_xy[idx] = ema_update(smoothed_lh_norm_xy[idx], (float(lm.x), float(lm.y)), alpha)
                        valid_lh_mask[idx] = True
                else:
                    valid_lh_mask = [pt is not None for pt in smoothed_lh_norm_xy]
                # Right hand
                if getattr(results, "right_hand_landmarks", None):
                    for idx, lm in enumerate(results.right_hand_landmarks.landmark[:rh_count]):
                        smoothed_rh_norm_xy[idx] = ema_update(smoothed_rh_norm_xy[idx], (float(lm.x), float(lm.y)), alpha)
                        valid_rh_mask[idx] = True
                else:
                    valid_rh_mask = [pt is not None for pt in smoothed_rh_norm_xy]

            # Face mesh (optional or required in dense mode)
            if use_holistic_needed and (enable_face or current_pose_density == "dense"):
                if getattr(results, "face_landmarks", None):
                    for idx, lm in enumerate(results.face_landmarks.landmark[:face_count]):
                        smoothed_face_norm_xy[idx] = ema_update(smoothed_face_norm_xy[idx], (float(lm.x), float(lm.y)), alpha)
                        valid_face_mask[idx] = True
                else:
                    valid_face_mask = [pt is not None for pt in smoothed_face_norm_xy]

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

            # Optional extra pixel coordinates
            lh_points_px: List[Optional[Tuple[int, int]]] = [None] * lh_count
            rh_points_px: List[Optional[Tuple[int, int]]] = [None] * rh_count
            face_points_px: List[Optional[Tuple[int, int]]] = [None] * face_count
            if use_holistic_needed and enable_hands:
                for i, sxy in enumerate(smoothed_lh_norm_xy):
                    if sxy is None:
                        continue
                    lh_points_px[i] = (int(round(sxy[0] * w)), int(round(sxy[1] * h)))
                for i, sxy in enumerate(smoothed_rh_norm_xy):
                    if sxy is None:
                        continue
                    rh_points_px[i] = (int(round(sxy[0] * w)), int(round(sxy[1] * h)))
            if use_holistic_needed and (enable_face or current_pose_density == "dense"):
                # Draw a subset for performance – still compute px for consistency
                for i, sxy in enumerate(smoothed_face_norm_xy):
                    if sxy is None:
                        continue
                    face_points_px[i] = (int(round(sxy[0] * w)), int(round(sxy[1] * h)))

            # Keep a raw copy for side-by-side display if requested
            raw_view = frame_bgr.copy()

            # Draw smoothed skeleton with selected density
            # SPARSE uses our augmented set (current default body), DENSE adds face tessellation
            pose_connections = get_pose_connections(dense=(current_pose_density == "sparse"))
            draw_smoothed_skeleton(frame_bgr, points_px, valid_mask, pose_connections)
            # Hands (draw full hand connectivity)
            if use_holistic_needed and enable_hands:
                mp_hands = mp.solutions.hands
                draw_smoothed_skeleton(frame_bgr, lh_points_px, valid_lh_mask, mp_hands.HAND_CONNECTIONS)
                draw_smoothed_skeleton(frame_bgr, rh_points_px, valid_rh_mask, mp_hands.HAND_CONNECTIONS)
                # Extra dense vectors on hands when in dense mode
                if current_pose_density == "dense":
                    draw_hand_dense_vectors(frame_bgr, lh_points_px, valid_lh_mask)
                    draw_hand_dense_vectors(frame_bgr, rh_points_px, valid_rh_mask)
                if gesture_engine is not None:
                    gesture_engine.update_hand_points(lh_points_px, rh_points_px)
            # Face mesh: in DENSE draw tessellation, otherwise draw contours if face enabled
            if use_holistic_needed and (enable_face or current_pose_density == "dense"):
                if current_pose_density == "dense":
                    fm_connections = getattr(getattr(mp.solutions, "holistic", None), "FACEMESH_TESSELATION", None)
                    if fm_connections is None:
                        fm_connections = getattr(getattr(mp.solutions, "face_mesh", None), "FACEMESH_TESSELATION", [])
                else:
                    fm_connections = getattr(getattr(mp.solutions, "holistic", None), "FACEMESH_CONTOURS", None)
                    if fm_connections is None:
                        fm_connections = getattr(getattr(mp.solutions, "face_mesh", None), "FACEMESH_CONTOURS", [])
                if fm_connections:
                    draw_smoothed_skeleton(frame_bgr, face_points_px, valid_face_mask, fm_connections)

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

            # Movement-specific FSM updates and overlays
            primary_knee_angle: Optional[float] = None
            current_state: Optional[str] = None
            current_reps: Optional[int] = None
            last_feedback: Optional[str] = None
            overlay_border_color = (0, 255, 255)  # default: yellow

            if squat_fsm is not None:
                # Choose the deeper knee (smaller angle)
                lk = angles.get("left_knee")
                rk = angles.get("right_knee")
                side = None
                if lk is not None and rk is not None:
                    if lk <= rk:
                        primary_knee_angle = lk
                        side = "left"
                    else:
                        primary_knee_angle = rk
                        side = "right"
                elif lk is not None:
                    primary_knee_angle = lk
                    side = "left"
                elif rk is not None:
                    primary_knee_angle = rk
                    side = "right"

                hip_y_px: Optional[int] = None
                knee_y_px: Optional[int] = None
                if side is not None:
                    pl = mp.solutions.pose.PoseLandmark
                    if side == "left":
                        hip_idx = pl.LEFT_HIP.value
                        knee_idx = pl.LEFT_KNEE.value
                    else:
                        hip_idx = pl.RIGHT_HIP.value
                        knee_idx = pl.RIGHT_KNEE.value
                    if 0 <= hip_idx < len(points_px):
                        hip_pt = points_px[hip_idx]
                        if hip_pt is not None:
                            hip_y_px = hip_pt[1]
                    if 0 <= knee_idx < len(points_px):
                        knee_pt = points_px[knee_idx]
                        if knee_pt is not None:
                            knee_y_px = knee_pt[1]

                now_s = time.perf_counter()
                squat_fsm.update(now_s, primary_knee_angle, hip_y_px, knee_y_px)

                current_state = squat_fsm.state
                current_reps = squat_fsm.rep_count
                last_feedback = squat_fsm.last_feedback

                # Overlay texts
                put_overlay_text(frame_bgr, f"Squat State: {current_state}", origin=(10, 60))
                put_overlay_text(frame_bgr, f"Reps: {current_reps}", origin=(10, 90))
                if primary_knee_angle is not None:
                    put_overlay_text(frame_bgr, f"Knee Angle: {primary_knee_angle:5.1f}", origin=(10, 120))
                if last_feedback:
                    put_overlay_text(frame_bgr, f"Feedback: {last_feedback}", origin=(10, 150))

                # Border color cues
                if last_feedback in ("Too fast", "Too shallow", "Incomplete"):
                    overlay_border_color = (0, 0, 255)  # red
                elif last_feedback == "OK":
                    overlay_border_color = (0, 255, 0)  # green
                else:
                    overlay_border_color = (0, 255, 255)  # yellow

                # Draw border
                h_b, w_b = frame_bgr.shape[:2]
                _cv2_call("rectangle", frame_bgr, (0, 0), (w_b - 1, h_b - 1), overlay_border_color, 3)

            # Movement-specific FSM updates and overlays
            timestamp_ms = int((frame_end_time - start_monotonic) * 1000)
            keypoints_list = []
            for i in range(num_landmarks):
                sxy = smoothed_norm_xy[i]
                conf = measurement_conf[i]
                if sxy is None:
                    keypoints_list.append({"x": None, "y": None, "confidence": conf})
                else:
                    keypoints_list.append({"x": sxy[0], "y": sxy[1], "confidence": conf})

            frame_record = {
                    "timestamp_ms": timestamp_ms,
                    "keypoints": keypoints_list,
                    "angles": angles,
            }
            if squat_fsm is not None:
                frame_record.update(
                    {
                        "squat_state": current_state,
                        "squat_reps": current_reps,
                        "primary_knee_angle": primary_knee_angle,
                        "feedback": last_feedback,
                    }
                )
            # Action recognition
            if action_recognizer is not None:
                action_recognizer.update(time.perf_counter(), smoothed_norm_xy, points_px)
                put_overlay_text(frame_bgr, f"Action: {action_recognizer.current_label} ({action_recognizer.current_confidence:.2f})", origin=(10, 180))
                frame_record["action_label"] = action_recognizer.current_label
                frame_record["action_confidence"] = action_recognizer.current_confidence

            # Gesture engine: map gestures to actions (e.g., next/prev slide, pause)
            if gesture_engine is not None:
                fired = gesture_engine.step(points_px, smoothed_norm_xy)
                if fired:
                    put_overlay_text(frame_bgr, f"Gestures: {', '.join(fired)}", origin=(10, 210))
                    frame_record["gestures"] = fired

            # Show current pose density label
            put_overlay_text(frame_bgr, f"Pose: {current_pose_density.upper()}", origin=(10, 240))

            analytics_frames.append(frame_record)

            # Show views
            if side_by_side:
                combined = _concat_h(raw_view, frame_bgr)
                _cv2_call("imshow", "Pose: Raw | Analytics (Press 'q' to quit)", combined)
            else:
                _cv2_call("imshow", "Baseline Pose (Press 'q' to quit)", frame_bgr)
            key = _cv2_call("waitKey", 1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                current_pose_density = "dense"
            elif key == ord('s'):
                current_pose_density = "sparse"

    finally:
        cap.release()
        _cv2_call("destroyAllWindows")
        try:
            estimator.close()
        except (AttributeError, RuntimeError):
            pass

        # Export analytics if requested
        if export_json:
            try:
                with open(export_json, "w", encoding="utf-8") as f:
                    json.dump(analytics_frames, f)
            except (OSError, IOError, ValueError, TypeError) as e:
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
            except (OSError, IOError, ValueError, TypeError) as e:
                print(f"Failed to write CSV export to {export_csv}: {e}")

        # Print summary metrics to console
        if squat_fsm is not None:
            summary = squat_fsm.summary_metrics()
            print(
                (
                    f"\nSquat Summary: reps={int(summary['reps'] or 0)} | "
                    f"avg_depth_angle={summary['avg_depth_angle']:.1f} deg "
                    f"| avg_rep_time={summary['avg_rep_time_s']:.2f}s "
                    f"| std_rep_time={summary['std_rep_time_s']:.2f}s "
                    f"| too_fast={int(summary['too_fast_count'] or 0)}\n"
                )
                if summary["avg_depth_angle"] is not None and summary["avg_rep_time_s"] is not None
                else f"\nSquat Summary: reps={int(summary['reps'] or 0)} | insufficient data for averages\n"
            )


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
    parser.add_argument(
        "--movement",
        type=str,
        default="none",
        choices=["none", "squat"],
        help="Targeted movement analysis (default: none). Use 'squat' to enable squat FSM.",
    )
    parser.add_argument("--side_by_side", action="store_true", help="Show raw and analytics views side-by-side")
    parser.add_argument(
        "--squat_bottom_angle",
        type=float,
        default=80.0,
        help="Knee angle threshold (deg) to consider squat bottom reached (default: 80)",
    )
    parser.add_argument(
        "--squat_top_angle",
        type=float,
        default=160.0,
        help="Knee angle threshold (deg) to consider top position (default: 160)",
    )
    parser.add_argument(
        "--hip_knee_delta_px",
        type=int,
        default=20,
        help="Required vertical pixel delta (hip_y - knee_y) at bottom (default: 20)",
    )
    parser.add_argument(
        "--tempo_min_s",
        type=float,
        default=0.5,
        help="Minimum seconds between bottoms; faster is flagged as too fast (default: 0.5)",
    )
    # Actions on by default; allow disabling with --no_actions
    parser.add_argument("--no_actions", dest="enable_actions", action="store_false", help="Disable heuristic action recognition overlays")
    parser.set_defaults(enable_actions=True)
    parser.add_argument(
        "--action_window_s",
        type=float,
        default=1.5,
        help="Sliding window size in seconds for action recognition (default: 1.5)",
    )
    parser.add_argument("--enable_hands", action="store_true", help="Add 21-point hand landmarks and connections (all finger joints)")
    parser.add_argument("--enable_face", action="store_true", help="Add 468-point face mesh contours; enabling this refines face landmarks for better head shape")
    parser.add_argument("--gesture_min_frames", type=int, default=5, help="Frames required to confirm a gesture (temporal debounce)")
    parser.add_argument("--gesture_cooldown_frames", type=int, default=20, help="Cooldown frames after a gesture fires")
    parser.add_argument(
        "--pose_density",
        type=str,
        choices=["dense", "sparse"],
        default="dense",
        help="Pose connection density for drawing (default: dense)",
    )
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
        movement=args.movement,
        side_by_side=args.side_by_side,
        squat_bottom_angle=args.squat_bottom_angle,
        squat_top_angle=args.squat_top_angle,
        hip_knee_delta_px=args.hip_knee_delta_px,
        tempo_min_s=args.tempo_min_s,
        enable_actions=args.enable_actions,
        action_window_s=args.action_window_s,
        enable_hands=args.enable_hands,
        enable_face=args.enable_face,
        gesture_min_frames=args.gesture_min_frames,
        gesture_cooldown_frames=args.gesture_cooldown_frames,
        pose_density=args.pose_density,
    )


