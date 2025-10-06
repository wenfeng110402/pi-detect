# Pi Detect

[![Hackatime Stats](https://hackatime-badge.hackclub.com/U09JFS1BU2V/pi-detect)](https://hackatime-badge.hackclub.com/U09JFS1BU2V/pi-detect)

## Project Introduction

Pi Detect is a lightweight human detection system designed specifically for Raspberry Pi that can perform real-time human recognition and counting functions on resource-constrained edge devices. This project combines advanced YOLO object detection models with optimized image processing techniques to provide users with accurate and efficient detection experience.

## Features

- 🎯 **Real-time Human Detection** - High-precision human recognition based on YOLOv8 model
- 🚀 **Lightweight Design** - Optimized for edge devices like Raspberry Pi with low resource consumption
- 🖥️ **User-friendly Interface** - Modern Fluent UI design for simple and intuitive operation
- ⚙️ **Multi-source Input Support** - Supports camera, local files, and network URL input
- 🎛️ **Adjustable Parameters** - Supports adjustment of confidence threshold, frame skipping, and image size
- 📊 **Real-time Statistics** - Provides real-time statistics of detected people

## Tech Stack

- **Core Framework**: [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- **Image Processing**: [OpenCV](https://opencv.org/)
- **UI Design**: [PyQt6](https://pypi.org/project/PyQt6/) + [PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)
- **Numerical Computing**: [NumPy](https://numpy.org/)

## System Requirements

### Hardware Requirements

- Raspberry Pi 4B/5 (4GB RAM or more recommended)
- Compatible camera module (Pi Camera or USB Webcam)
- Cooling device (recommended for long-term operation)

### Software Requirements

- Raspberry Pi OS (64-bit recommended)
- Python 3.7+
- pip

## Installation Guide

1. Clone the project locally:

   ```bash
   git clone <repository-url>
   cd pi-detect
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the program:

   ```bash
   # Run the GUI version
   python gui.py

   # Run the lightweight command-line version (optimized for Raspberry Pi)
   python lite_ver_forRpi.py
   ```

## Usage Instructions

### GUI Version (gui.py)

- Provides complete visual interface and parameter adjustment functions
- Supports switching between multiple input sources (camera, local files, network streams)
- Displays detection results and people count in real-time
- Adjustable parameters including confidence threshold, frame skipping, etc.

### Command-line Version (lite_ver_forRpi.py)

```bash
python lite_ver_forRpi.py --source 0 --imgsz 416 --skip 2 --conf 0.3
```

Parameter descriptions:
- `--source`: Input source (default: 0, which means default camera)
- `--imgsz`: Inference image size (default: 416)
- `--skip`: Frame skipping, process 1 frame out of every N frames (default: 2)
- `--conf`: Confidence threshold (default: 0.3)
- `--show`: Display detection window
- `--device`: Inference device (default: cpu)

## Project Structure

```
pi-detect/
├── gui.py                 # GUI main program
├── lite_ver_forRpi.py     # Lightweight command-line version
├── pyqt6_tutorial.py      # PyQt6 tutorial
├── requirements.txt       # Project dependencies file
├── README.md             # Project documentation (English)
└── README_zh.md          # Project documentation (Chinese)
```

## Development Background

This project originated from preliminary work in a science and technology innovation competition, where the basic function of using OpenCV to call models was completed. In the later stages, after in-depth design and optimization, the UI part took nearly 20 hours, evolving from a command-line interface to tkinter, PyQt, and finally adopting the modern Fluent UI design language, providing a more beautiful and user-friendly interface.

## License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for details.
# Pi Detect

[![Hackatime Stats](https://hackatime-badge.hackclub.com/U09JFS1BU2V/pi-detect)](https://hackatime-badge.hackclub.com/U09JFS1BU2V/pi-detect)

## Project Introduction

Pi Detect is a lightweight human detection system designed specifically for Raspberry Pi that can perform real-time human recognition and counting functions on resource-constrained edge devices. This project combines advanced YOLO object detection models with optimized image processing techniques to provide users with accurate and efficient detection experience.

## Features

- 🎯 **Real-time Human Detection** - High-precision human recognition based on YOLOv8 model
- 🚀 **Lightweight Design** - Optimized for edge devices like Raspberry Pi with low resource consumption
- 🖥️ **User-friendly Interface** - Modern Fluent UI design for simple and intuitive operation
- ⚙️ **Multi-source Input Support** - Supports camera, local files, and network URL input
- 🎛️ **Adjustable Parameters** - Supports adjustment of confidence threshold, frame skipping, and image size
- 📊 **Real-time Statistics** - Provides real-time statistics of detected people

## Tech Stack

- **Core Framework**: [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- **Image Processing**: [OpenCV](https://opencv.org/)
- **UI Design**: [PyQt6](https://pypi.org/project/PyQt6/) + [PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)
- **Numerical Computing**: [NumPy](https://numpy.org/)

## System Requirements

### Hardware Requirements

- Raspberry Pi 4B/5 (4GB RAM or more recommended)
- Compatible camera module (Pi Camera or USB Webcam)
- Cooling device (recommended for long-term operation)

### Software Requirements

- Raspberry Pi OS (64-bit recommended)
- Python 3.7+
- pip

## Installation Guide

1. Clone the project locally:

   ```bash
   git clone <repository-url>
   cd pi-detect
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the program:

   ```bash
   # Run the GUI version
   python gui.py

   # Run the lightweight command-line version (optimized for Raspberry Pi)
   python lite_ver_forRpi.py
   ```

## Usage Instructions

### GUI Version (gui.py)

- Provides complete visual interface and parameter adjustment functions
- Supports switching between multiple input sources (camera, local files, network streams)
- Displays detection results and people count in real-time
- Adjustable parameters including confidence threshold, frame skipping, etc.

### Command-line Version (lite_ver_forRpi.py)

```bash
python lite_ver_forRpi.py --source 0 --imgsz 416 --skip 2 --conf 0.3
```

Parameter descriptions:
- `--source`: Input source (default: 0, which means default camera)
- `--imgsz`: Inference image size (default: 416)
- `--skip`: Frame skipping, process 1 frame out of every N frames (default: 2)
- `--conf`: Confidence threshold (default: 0.3)
- `--show`: Display detection window
- `--device`: Inference device (default: cpu)

## Project Structure

```
pi-detect/
├── gui.py                 # GUI main program
├── lite_ver_forRpi.py     # Lightweight command-line version
├── pyqt6_tutorial.py      # PyQt6 tutorial
├── requirements.txt       # Project dependencies file
├── README.md             # Project documentation (English)
└── README_zh.md          # Project documentation (Chinese)
```

## Development Background

This project originated from preliminary work in a science and technology innovation competition, where the basic function of using OpenCV to call models was completed. In the later stages, after in-depth design and optimization, the UI part took nearly 20 hours, evolving from a command-line interface to tkinter, PyQt, and finally adopting the modern Fluent UI design language, providing a more beautiful and user-friendly interface.

## License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for details.