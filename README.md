# pi-detect (Hackberry project)

A lightweight person-detection project that runs smoothly on a Raspberry Pi using a YOLO model.
This project is designed to detect people in a camera feed with minimal hardware requirements, making
it ideal for monitoring and managing the number of people in a defined area (rooms, entryways, small venues).

## Key features

- Real-time person detection on Raspberry Pi hardware.
- Uses a compact YOLO model optimized for edge devices to balance accuracy and speed.
- Simple setup intended to run with only a Raspberry Pi and a compatible camera (USB or Pi Camera).
- Outputs bounding boxes and person counts suitable for area monitoring and simple analytics.

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
