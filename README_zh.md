# Pi Detect

[![Hackatime Stats](https://hackatime-badge.hackclub.com/U09JFS1BU2V/pi-detect)](https://hackatime-badge.hackclub.com/U09JFS1BU2V/pi-detect)

## 项目简介

Pi Detect 是一个专为树莓派设计的轻量化人体检测系统，能够在资源受限的边缘设备上实现实时人体识别和计数功能。该项目结合了先进的YOLO目标检测模型和优化的图像处理技术，为用户提供准确、高效的检测体验。

## 功能特点

- 🎯 **实时人体检测** - 基于YOLOv8模型实现高精度人体识别
- 🚀 **轻量化设计** - 专为树莓派等边缘设备优化，资源占用低
- 🖥️ **友好用户界面** - 采用现代化Fluent UI设计，操作简单直观
- ⚙️ **多源输入支持** - 支持摄像头、本地文件和网络URL输入
- 🎛️ **参数可调** - 支持置信度阈值、跳帧数和图像尺寸调节
- 📊 **实时统计** - 提供检测到的人数实时统计功能

## 技术栈

- **核心框架**: [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- **图像处理**: [OpenCV](https://opencv.org/)
- **界面设计**: [PyQt6](https://pypi.org/project/PyQt6/) + [PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)
- **数值计算**: [NumPy](https://numpy.org/)

## 系统要求

### 硬件要求

- 树莓派 4B/5 (推荐4GB RAM以上)
- 兼容的摄像头模块 (Pi Camera 或 USB Webcam)
- 散热装置 (推荐，长时间运行使用)

### 软件要求

- Raspberry Pi OS (64-bit recommended)
- Python 3.7+
- pip

## 安装指南

1. 克隆项目到本地:

   ```bash
   git clone <repository-url>
   cd pi-detect
   ```
2. 安装依赖:

   ```bash
   pip install -r requirements.txt
   ```
3. 运行程序:

   ```bash
   # 运行图形界面版本
   python gui.py

   # 运行轻量级命令行版本（针对树莓派优化）
   python lite_ver_forRpi.py
   ```

## 使用说明

### 图形界面版本 (gui.py)

- 提供完整的可视化界面和参数调节功能
- 支持多种输入源切换（摄像头、本地文件、网络流）
- 实时显示检测结果和人数统计
- 可调节置信度阈值、跳帧数等参数

### 命令行版本 (lite_ver_forRpi.py)

```bash
python lite_ver_forRpi.py --source 0 --imgsz 416 --skip 2 --conf 0.3
```
