# Aircraft Runway Occupancy Detector

Detects and segments aircraft presence episodes in runway surveillance video using Faster R-CNN (+ SAHI) and a Kalman-filtered state machine.

## Install

```bash
pip install -r requirements.txt
```

> Requires CUDA. PyTorch is installed with CUDA 11.8 by default — adjust the index URL in `requirements.txt` if needed.

## Run

```bash
python src/main.py --config config/default.yaml --source /path/to/video.mp4
```

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config file |
| `--source` | Path to input video |
| `--verbose` | Print per-frame state machine debug output |

## Configure

Edit `config/default.yaml`. Key settings:

```yaml
frame_stride: 4          # sample every Nth frame
use_sahi: true           # enable SAHI slicing for small objects
roi:                     # polygon ROI filter (null = full frame)
  - [1150, 1200]
  - [7500, 770]
  - [7500, 0]
  - [1000, 0]

episode_state_machine:
  start_conf: 0.5        # min confidence to trigger episode start
  gate_px: 500           # max px from Kalman prediction to accept detection
  lost_timeout: 20       # sampled frames of missed detection before episode closes
```

## Outputs

Results are saved to `data/<video_name>/`:

| File | Description |
|------|-------------|
| `presence.csv` | Per-frame: `time_sec, presence, episode_id` |
| `presence_episodes.png` | Episode chart with coloured bands |
| `images/` | Annotated frames (if `save_img: true`) |
| `detection_video.mp4` | Video from annotated frames (if `generate_video: true`) |

## Supported Models

Set `weights` in config to one of:

- `fasterrcnn_resnet50_fpn`
- `fasterrcnn_resnet50_fpn_v2`
- `retinanet_resnet50_fpn`
