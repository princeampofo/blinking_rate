"""
Part B: Facial Dimension Estimator

Usage:
  python dimensions.py --video path/to/video.mp4
  python dimensions.py --video path/to/video.mp4 --reference_mm 63 --output_annotated annotated.mp4

Dependencies:
  pip install opencv-python numpy matplotlib mediapipe==0.10.9
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from scipy.spatial import distance as dist


# Eyes
LEFT_EYE_OUTER  = 263   # outer corner (left eye, from viewer's right)
LEFT_EYE_INNER  = 362   # inner corner
RIGHT_EYE_OUTER = 33    # outer corner (right eye, from viewer's left)
RIGHT_EYE_INNER = 133   # inner corner

# Interpupillary reference landmarks
LEFT_PUPIL  = 468       # (refined landmark, needs refine_landmarks=True)
RIGHT_PUPIL = 473

# Face extents
FACE_TOP    = 10        # top of forehead (approx)
FACE_CHIN   = 152       # chin bottom
FACE_LEFT   = 234       # left cheek / ear edge
FACE_RIGHT  = 454       # right cheek / ear edge

# Nose
NOSE_TIP     = 4
NOSE_BRIDGE  = 6        # bridge (between eyes)
NOSE_LEFT_NOSTRIL  = 279
NOSE_RIGHT_NOSTRIL = 49

# Mouth
MOUTH_LEFT   = 61       # left corner
MOUTH_RIGHT  = 291      # right corner
MOUTH_TOP    = 13       # upper lip center
MOUTH_BOTTOM = 14       # lower lip center


def euclidean(lms, i, j, w, h):
    """Euclidean pixel distance between landmark i and j."""
    p1 = np.array([lms[i].x * w, lms[i].y * h])
    p2 = np.array([lms[j].x * w, lms[j].y * h])
    return np.linalg.norm(p1 - p2)


def measure_frame(lms, img_w, img_h):
    """
    Given face mesh landmarks for one frame, return a dict of pixel distances.
    """
    m = {}

    # Eyes (corner-to-corner width)
    m["left_eye_width"]  = euclidean(lms, LEFT_EYE_OUTER,  LEFT_EYE_INNER,  img_w, img_h)
    m["right_eye_width"] = euclidean(lms, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, img_w, img_h)

    # Interpupillary distance (IPD) - used as real-world reference if known
    m["ipd"] = euclidean(lms, LEFT_PUPIL, RIGHT_PUPIL, img_w, img_h)

    # Face height (top of head to chin)
    m["face_height"] = euclidean(lms, FACE_TOP, FACE_CHIN, img_w, img_h)

    # Face width (ear to ear)
    m["face_width"]  = euclidean(lms, FACE_LEFT, FACE_RIGHT, img_w, img_h)

    # Nose
    m["nose_width"]  = euclidean(lms, NOSE_LEFT_NOSTRIL, NOSE_RIGHT_NOSTRIL, img_w, img_h)
    m["nose_height"] = euclidean(lms, NOSE_BRIDGE, NOSE_TIP, img_w, img_h)

    # Mouth
    m["mouth_width"]  = euclidean(lms, MOUTH_LEFT,   MOUTH_RIGHT,  img_w, img_h)
    m["mouth_height"] = euclidean(lms, MOUTH_TOP,    MOUTH_BOTTOM, img_w, img_h)

    return m


def draw_annotations(frame, lms, img_w, img_h, measurements):
    """Draw measurement lines and labels onto a frame."""
    def pt(idx):
        return (int(lms[idx].x * img_w), int(lms[idx].y * img_h))

    def line(i, j, color, label, offset=(0, -8)):
        p1, p2 = pt(i), pt(j)
        cv2.line(frame, p1, p2, color, 2)
        mid = ((p1[0]+p2[0])//2 + offset[0], (p1[1]+p2[1])//2 + offset[1])
        cv2.putText(frame, label, mid, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)

    line(LEFT_EYE_OUTER, LEFT_EYE_INNER, (255,200,0), f"L-eye {measurements['left_eye_width']:.0f}px")
    line(RIGHT_EYE_OUTER, RIGHT_EYE_INNER, (255,200,0), f"R-eye {measurements['right_eye_width']:.0f}px")
    line(FACE_TOP, FACE_CHIN, (0,255,100), f"H {measurements['face_height']:.0f}px", offset=(-60,0))
    line(FACE_LEFT, FACE_RIGHT, (0,180,255), f"W {measurements['face_width']:.0f}px", offset=(0,-10))
    line(NOSE_LEFT_NOSTRIL, NOSE_RIGHT_NOSTRIL, (255,100,255), f"NW {measurements['nose_width']:.0f}px", offset=(0,14))
    line(NOSE_BRIDGE, NOSE_TIP, (255,100,255), f"NH {measurements['nose_height']:.0f}px", offset=(8,0))
    line(MOUTH_LEFT, MOUTH_RIGHT, (0,220,220), f"MW {measurements['mouth_width']:.0f}px", offset=(0,14))
    line(MOUTH_TOP, MOUTH_BOTTOM, (0,220,220), f"MH {measurements['mouth_height']:.0f}px", offset=(8,0))
    return frame


def run_dimension_estimation(video_path: str,
                              reference_mm: float = None,
                              known_ref_key: str = "ipd",
                              output_annotated: str = None,
                              sample_every_n: int = 5,
                              output_plot: str = "face_dims_plot.png"):
    """
    Main pipeline for facial dimension estimation.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video    : {video_path}")
    print(f"[INFO] Size     : {fw}x{fh}  |  FPS: {fps:.1f}  |  Frames: {total}")
    print(f"[INFO] Sampling : every {sample_every_n} frames")

    writer = None
    if output_annotated:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_annotated, fourcc, fps, (fw, fh))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,       # needed for pupil landmarks 468 & 473
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    all_measurements = defaultdict(list)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n == 0:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                m   = measure_frame(lms, fw, fh)
                for k, v in m.items():
                    all_measurements[k].append(v)

                if output_annotated and writer:
                    frame = draw_annotations(frame, lms, fw, fh, m)

        if output_annotated and writer:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    face_mesh.close()
    if writer:
        writer.release()

    # Aggregate statistics
    stats = {}
    for k, vals in all_measurements.items():
        arr = np.array(vals)
        stats[k] = {
            "median_px": np.median(arr),
            "mean_px":   np.mean(arr),
            "std_px":    np.std(arr),
            "n_frames":  len(arr),
        }

    # Convert to mm if reference provided
    scale_px_to_mm = None
    if reference_mm and known_ref_key in stats:
        ref_px         = stats[known_ref_key]["median_px"]
        scale_px_to_mm = reference_mm / ref_px
        print(f"\n[INFO] Real-world scale: {scale_px_to_mm:.4f} mm/px "
              f"  (based on {known_ref_key} = {reference_mm} mm)")

    # Print report
    LABELS = {
        "left_eye_width":  "Left Eye Width",
        "right_eye_width": "Right Eye Width",
        "ipd":             "Interpupillary Dist.",
        "face_height":     "Face Height (top→chin)",
        "face_width":      "Face Width (ear→ear)",
        "nose_width":      "Nose Width (nostril→nostril)",
        "nose_height":     "Nose Height (bridge→tip)",
        "mouth_width":     "Mouth Width",
        "mouth_height":    "Mouth Height (lip gap)",
    }

    print("\n" + "="*68)
    print(f"  FACIAL DIMENSION RESULTS  (sampled from {frame_idx} frames)")
    print("="*68)
    header = f"  {'Feature':<30} {'Median (px)':>12} {'Std (px)':>10}"
    if scale_px_to_mm:
        header += f" {'Median (mm)':>12}"
    print(header)
    print("-"*68)

    results_dict = {}
    for key, label in LABELS.items():
        if key not in stats:
            continue
        s  = stats[key]
        mm = s["median_px"] * scale_px_to_mm if scale_px_to_mm else None
        row = f"  {label:<30} {s['median_px']:>12.1f} {s['std_px']:>10.1f}"
        if mm is not None:
            row += f" {mm:>12.1f}"
        print(row)
        results_dict[label] = {"median_px": s["median_px"], "median_mm": mm}

    print("="*68)

    # Bar chart
    labels_plot = list(results_dict.keys())
    px_vals     = [results_dict[l]["median_px"] for l in labels_plot]
    mm_vals     = [results_dict[l]["median_mm"] for l in labels_plot] if scale_px_to_mm else None

    fig, axes = plt.subplots(1, 2 if mm_vals else 1, figsize=(14, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    y_pos = np.arange(len(labels_plot))
    axes[0].barh(y_pos, px_vals, color="#2196F3", alpha=0.85)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels_plot, fontsize=9)
    axes[0].set_xlabel("Pixels")
    axes[0].set_title("Facial Dimensions (pixels)")
    for i, v in enumerate(px_vals):
        axes[0].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)

    if mm_vals and len(axes) > 1:
        axes[1].barh(y_pos, mm_vals, color="#4CAF50", alpha=0.85)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(labels_plot, fontsize=9)
        axes[1].set_xlabel("Millimetres")
        axes[1].set_title(f"Facial Dimensions (mm)  [ref: {known_ref_key}={reference_mm}mm]")
        for i, v in enumerate(mm_vals):
            if v:
                axes[1].text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=8)

    plt.suptitle("Estimated Facial Dimensions from Video", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"[INFO] Bar chart saved → {output_plot}")
    plt.show()

    return results_dict


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Dimension Estimator (Part B)")
    parser.add_argument("--video",            required=True,
                        help="Path to input video file")
    parser.add_argument("--reference_mm",     type=float, default=None,
                        help="Known real-world size of reference feature in mm "
                             "(e.g. --reference_mm 63 for average IPD)")
    parser.add_argument("--known_ref_key",    default="ipd",
                        choices=["ipd","face_width","face_height","nose_width","mouth_width"],
                        help="Which facial feature to use as real-world scale reference")
    parser.add_argument("--output_annotated", default=None,
                        help="(Optional) path to save annotated output video")
    parser.add_argument("--sample_every_n",   type=int, default=5,
                        help="Process every N-th frame (default: 5)")
    parser.add_argument("--output_plot",      default="face_dims_plot.png",
                        help="Path to save the summary bar chart")
    args = parser.parse_args()

    run_dimension_estimation(
        video_path       = args.video,
        reference_mm     = args.reference_mm,
        known_ref_key    = args.known_ref_key,
        output_annotated = args.output_annotated,
        sample_every_n   = args.sample_every_n,
        output_plot      = args.output_plot,
    )