"""
Part A: Eye Blink Rate Estimator
=================================
Estimates the number of eye blinks per second from a video recording
using facial landmark detection (MediaPipe Face Mesh).

Method:
  - Compute the Eye Aspect Ratio (EAR) for both eyes per frame.
  - A blink is detected when EAR drops below a threshold and rises back.
  - Report total blinks and blinks-per-second (avg over full recording).

Usage:
  python part_a_blink_rate.py --video path/to/video.mp4
  python part_a_blink_rate.py --video path/to/video.mp4 --ear_threshold 0.20 --consec_frames 2

Dependencies:
  pip install opencv-python mediapipe numpy matplotlib
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.spatial import distance as dist


# ─── MediaPipe Face Mesh landmark indices for each eye ───────────────────────
# Left eye:  landmarks 362, 385, 387, 263, 373, 380
# Right eye: landmarks  33, 160, 158, 133, 153, 144
# Layout (per eye): [p1(outer), p2(top-outer), p3(top-inner),
#                    p4(inner),  p5(bot-inner), p6(bot-outer)]

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [ 33, 160, 158, 133, 153, 144]


def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """
    Compute Eye Aspect Ratio (EAR).

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    A value near 0.3 → eye open
    A value near 0.0 → eye closed (blink)
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))
    pts = np.array(pts, dtype=np.float64)

    # Vertical distances
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    # Horizontal distance
    C = dist.euclidean(pts[0], pts[3])

    ear = (A + B) / (2.0 * C)
    return ear


def run_blink_detection(video_path: str,
                        ear_threshold: float = 0.20,
                        consec_frames: int = 2,
                        output_plot: str = "blink_ear_plot.png"):
    """
    Main pipeline for blink detection.

    Parameters
    ----------
    video_path     : Path to input video file.
    ear_threshold  : EAR value below which a blink is detected.
    consec_frames  : Number of consecutive frames below threshold to confirm blink.
    output_plot    : Path to save the EAR-over-time plot.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Resolution : {int(cap.get(3))} x {int(cap.get(4))}")
    print(f"[INFO] FPS        : {fps:.2f}")
    print(f"[INFO] Duration   : {duration_s:.1f} s  ({duration_s/3600:.2f} hr)")
    print(f"[INFO] EAR thresh : {ear_threshold}  |  consec frames: {consec_frames}")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    blink_count   = 0
    below_thresh  = 0       # consecutive frames below EAR threshold
    ear_history   = []      # (frame_index, avg_ear)
    frame_idx     = 0
    blink_in_prog = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        avg_ear = None
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            left_ear  = eye_aspect_ratio(lms, LEFT_EYE,  img_w, img_h)
            right_ear = eye_aspect_ratio(lms, RIGHT_EYE, img_w, img_h)
            avg_ear   = (left_ear + right_ear) / 2.0

            # ── Blink state machine ──────────────────────────────────────────
            if avg_ear < ear_threshold:
                below_thresh += 1
                blink_in_prog = True
            else:
                if blink_in_prog and below_thresh >= consec_frames:
                    blink_count += 1
                below_thresh  = 0
                blink_in_prog = False

        ear_history.append((frame_idx, avg_ear))
        frame_idx += 1

        # Optional: live preview (comment out for headless/long runs)
        # cv2.putText(frame, f"EAR: {avg_ear:.2f}  Blinks: {blink_count}",
        #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        # cv2.imshow("Blink Detector", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    face_mesh.close()
    # cv2.destroyAllWindows()

    blinks_per_second = blink_count / duration_s if duration_s > 0 else 0

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  BLINK DETECTION RESULTS")
    print("="*50)
    print(f"  Total frames processed : {frame_idx}")
    print(f"  Frames with face found : {sum(1 for _, e in ear_history if e is not None)}")
    print(f"  Total blinks detected  : {blink_count}")
    print(f"  Video duration         : {duration_s:.2f} s")
    print(f"  Blinks per second      : {blinks_per_second:.4f}")
    print(f"  Blinks per minute      : {blinks_per_second * 60:.2f}")
    print("="*50)

    # ── Plot EAR over time ───────────────────────────────────────────────────
    valid = [(f / fps, e) for f, e in ear_history if e is not None]
    if valid:
        times, ears = zip(*valid)
        plt.figure(figsize=(14, 4))
        plt.plot(times, ears, linewidth=0.6, color="#2196F3", label="Avg EAR")
        plt.axhline(y=ear_threshold, color="#F44336", linestyle="--",
                    linewidth=1.2, label=f"Threshold ({ear_threshold})")
        plt.fill_between(times, ears, ear_threshold,
                         where=[e < ear_threshold for e in ears],
                         color="#F44336", alpha=0.25, label="Blink zones")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Eye Aspect Ratio (EAR)")
        plt.title(f"Eye Aspect Ratio Over Time  |  Total Blinks: {blink_count}  "
                  f"|  {blinks_per_second:.4f} blinks/sec")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_plot, dpi=150)
        print(f"[INFO] EAR plot saved → {output_plot}")
        plt.show()

    return {
        "total_blinks": blink_count,
        "duration_seconds": duration_s,
        "blinks_per_second": blinks_per_second,
        "blinks_per_minute": blinks_per_second * 60,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eye Blink Rate Estimator (Part A)")
    parser.add_argument("--video",          required=True,  help="Path to input video file")
    parser.add_argument("--ear_threshold",  type=float, default=0.20,
                        help="EAR threshold for blink detection (default: 0.20)")
    parser.add_argument("--consec_frames",  type=int,   default=2,
                        help="Consecutive frames below threshold to confirm blink (default: 2)")
    parser.add_argument("--output_plot",    default="blink_ear_plot.png",
                        help="Path to save the EAR plot image")
    args = parser.parse_args()

    run_blink_detection(
        video_path    = args.video,
        ear_threshold = args.ear_threshold,
        consec_frames = args.consec_frames,
        output_plot   = args.output_plot,
    )