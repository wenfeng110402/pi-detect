# pi-detect (Hackberry project)

A lightweight person-detection project that runs smoothly on a Raspberry Pi using a YOLO model.
This project is designed to detect people in a camera feed with minimal hardware requirements, making
it ideal for monitoring and managing the number of people in a defined area (rooms, entryways, small venues).

## Key features

- Real-time person detection on Raspberry Pi hardware.
- Uses a compact YOLO model optimized for edge devices to balance accuracy and speed.
- Simple setup intended to run with only a Raspberry Pi and a compatible camera (USB or Pi Camera).
- Outputs bounding boxes and person counts suitable for area monitoring and simple analytics.
- GUI interface for easy source selection and parameter tuning.

## Why this project

Many modern object detection systems require powerful GPUs. This project focuses on a practical
constraint: run a YOLO-based person detector on a Raspberry Pi alone (no external GPU). It targets use cases
like small-room occupancy monitoring, queue length estimation, and basic access control where low-cost,
low-power hardware is preferred.

## Hardware requirements

- Raspberry Pi 4 or Pi 5 (4GB or 8GB recommended for best performance).
- A camera: Raspberry Pi Camera Module or a USB webcam supported by the Pi.
- MicroSD card with enough storage for OS and model files (16GB+ recommended).
- Optional: a small fan or heatsink for sustained performance on Pi 4.

## Software requirements

- Raspberry Pi OS (32-bit or 64-bit) or another Debian-based distribution for Raspberry Pi.
- Python 3.11+

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd pi-detect
   ```

2. Install required dependencies:
   ```
   pip install ultralytics opencv-python numpy pillow requests
   ```

## Usage

You can use either the command-line interface or the graphical user interface:

### Command-line interface

```
python main.py [--source CAMERA_INDEX] [--imgsz IMAGE_SIZE] [--skip SKIP_FRAMES] [--conf CONFIDENCE] [--show] [--device DEVICE]
```

Example:
```
python main.py --source 0 --imgsz 416 --skip 2 --conf 0.3 --show
```

### Graphical User Interface

Run the GUI version:
```
python gui.py
```

In the GUI, you can:
- Select between camera, local file, or network URL as media source
- Adjust detection parameters (confidence threshold, skip frames, image size)
- Start and stop detection with ease

## Command-line Arguments

- `--source`: Camera source index (default: 0)
- `--imgsz`: Inference image size (default: 416)
- `--skip`: Process every Nth frame (default: 2)
- `--conf`: Confidence threshold (default: 0.3)
- `--show`: Show display window
- `--device`: Inference device (default: 'cpu')