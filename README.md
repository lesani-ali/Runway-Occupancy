# Aircraft Runway Occupancy Detector

Detects and segments aircraft presence episodes in runway surveillance video using Faster R-CNN (+ SAHI) and a Kalman-filtered state machine.

## Install

**1. Clone and install (editable):**
```bash
git clone git@github.com:lesani-ali/Runway-Occupancy.git
cd Runway-Occupancy
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
```

> The `--extra-index-url` flag pulls PyTorch with CUDA 11.8 support. Change `cu118` to match your CUDA version (e.g. `cu121` for CUDA 12.1). See [pytorch.org](https://pytorch.org/get-started/locally/) for the full list.

## Run

**Detection pipeline:**
```bash
air-track run -c config/default.yaml -s /path/to/video.mp4
```

**Convert MKV recording to MP4:**
```bash
mkv2mp4 -i recording.mkv
mkv2mp4 -i recording.mkv -o output.mp4   # custom output path
```

> `mkv2mp4` re-encodes the video to H.264/AAC and produces a `.mp4` file.
> Requires `ffmpeg` to be installed and on your `PATH`.

**Help:**
```bash
air-track --help
air-track run --help
```

## Configure

Edit `config/default.yaml`. Key settings:

```yaml
video:
  frame_stride: 4          # sample every Nth frame

model:
  weights: "fasterrcnn_resnet50_fpn"
  device: "cuda:0"
  use_sahi: true           # enable SAHI slicing for small/distant objects

roi:
  polygon:                 # null = full frame
    - [870, 1070]
    - [7200, 0]
    - [700, 0]
  mask_frame: false

tracking:
  start_conf: 0.5          # min confidence to open an episode
  gate_px: 500             # max px from Kalman prediction to accept detection
  lost_timeout: 30         # missed frames before episode closes
```

## Outputs

Results are saved to `data/<video_stem>/`:

| File | Description |
|------|-------------|
| `presence.csv` | Per-frame: `time_sec, presence, episode_id` |
| `presence_episodes.png` | Episode chart with per-episode coloured bands |
| `episodes.txt` | Episode summary table (start, end, ROT) |
| `images/` | Annotated frames (if `save_img: true`) |
| `detection_video.mp4` | Video from annotated frames (if `generate_video: true`) |

## Supported Models

Set `model.weights` in config to one of:

| Model | Speed | Accuracy |
|-------|-------|----------|
| `fasterrcnn_resnet50_fpn` | fast | good |
| `fasterrcnn_resnet50_fpn_v2` | medium | better |
| `retinanet_resnet50_fpn` | fast | good for small objects |
