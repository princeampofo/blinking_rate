# Blinking Rate and Facial Dimensions Estimation

Two tools that use **MediaPipe Face Mesh** to analyse facial features from video. The videos were in .lrv format and were converted to .mp4 using [ffmpeg](https://ffmpeg.org/):

## Files

| File | Description |
|------|-------------|
| `blinking.py` | Estimates eye blink rate (blinks/second) from a video |
| `dimensions.py` | Measures facial dimensions (eye width, face size, nose, mouth, IPD) from a video |

---

## Part A — Eye Blink Rate (`blinking.py`)

Detects blinks by computing the **Eye Aspect Ratio (EAR)** for both eyes on every frame. A blink is registered when the EAR drops below a threshold for a minimum number of consecutive frames.

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2\,\|p_1 - p_4\|}$$

Outputs total blink count, blinks-per-second, and an EAR-over-time plot.

### Usage

```bash
python blinking.py --video path/to/video.mp4

# Optional parameters
python blinking.py --video path/to/video.mp4 --ear_threshold 0.20 --consec_frames 2
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to input video file |
| `--ear_threshold` | `0.20` | EAR value below which a blink is detected |
| `--consec_frames` | `2` | Consecutive frames below threshold required to confirm a blink |

---

## Part B — Facial Dimensions (`dimensions.py`)

Measures pixel distances between facial landmarks per frame and optionally converts them to millimetres using a known **interpupillary distance (IPD)** as a real-world reference.

Measured dimensions:
- Left / right eye width
- Interpupillary distance (IPD)
- Face height (forehead to chin) and face width (ear to ear)
- Nose width and height
- Mouth width and height

Outputs per-frame measurements, averaged results, and an annotated video.

### Usage

```bash
python dimensions.py --video path/to/video.mp4

# Provide a real-world reference to get mm estimates
python dimensions.py --video path/to/video.mp4 --reference_mm 63 --output_annotated annotated.mp4
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to input video file |
| `--reference_mm` | optional | Known IPD in mm used to scale pixel distances to real-world units |
| `--output_annotated` | optional | Path to save the annotated output video |

---

## Dependencies

```bash
pip install opencv-python numpy matplotlib scipy mediapipe==0.10.9
```
