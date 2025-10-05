import sys
import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QPushButton, QFileDialog, QStackedWidget, QComboBox, QButtonGroup, 
                            QRadioButton, QGroupBox, QSpinBox)
from qfluentwidgets import NavigationInterface, NavigationItemPosition, FluentWindow
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (SubtitleLabel, PrimaryPushButton, Slider, SpinBox, ComboBox,
                           setTheme, Theme, TransparentToolButton)
from ultralytics import YOLO
import numpy as np


class PersonDetectionGUI(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pi-Detect Human Detection System")
        self.resize(1000, 700)
        
        # Initialize variables
        self.model = None
        self.cap = None
        self.is_detecting = False
        self.is_image_mode = False
        self.current_source = None
        self.conf_threshold = 0.3
        self.skip_frames = 2
        self.img_size = 416
        
        # Initialize source variables
        self.camera_index = 0
        self.file_path = ""
        self.url = ""
        
        # UI initialization flag
        self.ui_initialized = False
        
        # Load model
        self.load_model()
        
    def showEvent(self, event):
        # Delay UI initialization until window is first shown
        if not self.ui_initialized:
            self.init_ui()
            self.ui_initialized = True
        super().showEvent(event)
        
    def init_ui(self):
        # Create pages
        self.home_page = self.create_home_page()
        self.settings_page = self.create_settings_page()
        self.about_page = self.create_about_page()
        
        # Add pages to navigation interface
        self.addSubInterface(self.home_page, FIF.HOME, "Home")
        self.addSubInterface(self.settings_page, FIF.SETTING, "Settings")
        self.addSubInterface(self.about_page, FIF.INFO, "About", NavigationItemPosition.BOTTOM)
        
        # Connect signals and slots
        self.init_connections()
        
    def create_home_page(self):
        page = QWidget()
        page.setObjectName("home_page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = SubtitleLabel("Human Detection System")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Media source selection
        source_group = QGroupBox("Media Source")
        source_layout = QVBoxLayout(source_group)
        
        # Source type selection
        source_type_layout = QHBoxLayout()
        self.source_camera_radio = QRadioButton("Camera")
        self.source_file_radio = QRadioButton("Local File")
        self.source_url_radio = QRadioButton("Network URL")
        
        self.source_camera_radio.setChecked(True)
        
        self.source_group = QButtonGroup()
        self.source_group.addButton(self.source_camera_radio, 0)
        self.source_group.addButton(self.source_file_radio, 1)
        self.source_group.addButton(self.source_url_radio, 2)
        
        source_type_layout.addWidget(self.source_camera_radio)
        source_type_layout.addWidget(self.source_file_radio)
        source_type_layout.addWidget(self.source_url_radio)
        source_layout.addLayout(source_type_layout)
        
        # Input controls layout
        self.input_layout = QVBoxLayout()
        
        # Camera input
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera Index:"))
        self.camera_spinbox = QSpinBox()
        self.camera_spinbox.setRange(0, 10)
        self.camera_spinbox.setValue(0)
        camera_layout.addWidget(self.camera_spinbox)
        self.input_layout.addLayout(camera_layout)
        
        source_layout.addLayout(self.input_layout)
        layout.addWidget(source_group)
        
        # Video display area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("Video Display Area")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            background-color: black; 
            color: white; 
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        """)
        layout.addWidget(self.video_label)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.start_button = PrimaryPushButton("Start Detection", self)
        self.stop_button = PrimaryPushButton("Stop Detection", self)
        self.stop_button.setEnabled(False)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        layout.addLayout(control_layout)
        
        layout.addStretch()
        return page
        
    def create_settings_page(self):
        page = QWidget()
        page.setObjectName("settings_page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = SubtitleLabel("Settings")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Confidence threshold setting
        conf_group = QGroupBox("Confidence Threshold")
        conf_layout = QHBoxLayout(conf_group)
        conf_layout.addWidget(QLabel("Threshold:"))
        self.conf_slider = Slider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(30)
        self.conf_value_label = QLabel("0.30")
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        layout.addWidget(conf_group)
        
        # Skip frames setting
        skip_group = QGroupBox("Skip Frames")
        skip_layout = QHBoxLayout(skip_group)
        skip_layout.addWidget(QLabel("Skip Frames:"))
        self.skip_spinbox = QSpinBox()
        self.skip_spinbox.setRange(1, 10)
        self.skip_spinbox.setValue(2)
        skip_layout.addWidget(self.skip_spinbox)
        layout.addWidget(skip_group)
        
        # Image size setting
        size_group = QGroupBox("Image Size")
        size_layout = QHBoxLayout(size_group)
        size_layout.addWidget(QLabel("Size:"))
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(320, 1024)
        self.size_spinbox.setValue(416)
        self.size_spinbox.setSingleStep(32)
        size_layout.addWidget(self.size_spinbox)
        layout.addWidget(size_group)
        
        # Camera setting
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QHBoxLayout(camera_group)
        camera_layout.addWidget(QLabel("Default Camera Index:"))
        self.camera_index_spinbox = QSpinBox()
        self.camera_index_spinbox.setRange(0, 10)
        self.camera_index_spinbox.setValue(0)
        camera_layout.addWidget(self.camera_index_spinbox)
        layout.addWidget(camera_group)
        
        layout.addStretch()
        return page
        
    def create_about_page(self):
        page = QWidget()
        page.setObjectName("about_page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        
        title = SubtitleLabel("About Pi-Detect")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        description = QLabel("""
        <h2>Pi-Detect Human Detection System</h2>
        <p>This is a human detection system based on Raspberry Pi, using YOLO model for real-time human detection.</p>
        <p><b>Key Features:</b></p>
        <ul>
            <li>Supports camera, local file, and network URL as input sources</li>
            <li>Real-time human detection and counting</li>
            <li>Adjustable detection parameters</li>
            <li>User-friendly graphical interface</li>
        </ul>
        <p><b>Tech Stack:</b></p>
        <ul>
            <li>YOLOv8 object detection model</li>
            <li>PyQt6 for GUI</li>
            <li>OpenCV for image processing</li>
            <li>Fluent UI design</li>
        </ul>
        """)
        description.setWordWrap(True)
        layout.addWidget(description)
        layout.addStretch()
        
        return page
        
    def init_connections(self):
        # Source selection connections
        self.source_camera_radio.toggled.connect(self.on_source_changed)
        self.source_file_radio.toggled.connect(self.on_source_changed)
        self.source_url_radio.toggled.connect(self.on_source_changed)
        
        # Control button connections
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        
        # Settings page connections
        self.conf_slider.valueChanged.connect(self.on_conf_threshold_changed)
        self.skip_spinbox.valueChanged.connect(self.on_skip_frames_changed)
        self.size_spinbox.valueChanged.connect(self.on_img_size_changed)
        self.camera_index_spinbox.valueChanged.connect(self.on_camera_index_changed)
        
    def on_source_changed(self):
        # Clear current input layout widgets
        for i in reversed(range(self.input_layout.count())):
            item = self.input_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
            elif item.layout():
                # Remove layout items
                layout = item.layout()
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
        
        if self.source_camera_radio.isChecked():
            # Camera input
            camera_layout = QHBoxLayout()
            camera_layout.addWidget(QLabel("Camera Index:"))
            self.camera_index_spinbox = QSpinBox()
            self.camera_index_spinbox.setRange(0, 10)
            self.camera_index_spinbox.setValue(self.camera_index)
            camera_layout.addWidget(self.camera_index_spinbox)
            self.input_layout.addLayout(camera_layout)
            
        elif self.source_file_radio.isChecked():
            # File input
            file_layout = QHBoxLayout()
            file_layout.addWidget(QLabel("File Path:"))
            self.file_input = ComboBox()
            self.file_input.setEditable(True)
            self.file_input.addItems([
                "example.mp4",
                "sample.avi"
            ])
            self.file_browse_button = QPushButton("Browse...")
            self.file_browse_button.clicked.connect(self.browse_file)
            file_layout.addWidget(self.file_input)
            file_layout.addWidget(self.file_browse_button)
            self.input_layout.addLayout(file_layout)
            
        elif self.source_url_radio.isChecked():
            # URL input
            url_layout = QHBoxLayout()
            url_layout.addWidget(QLabel("URL:"))
            self.url_input = ComboBox()
            self.url_input.setEditable(True)
            self.url_input.addItems([
                "http://example.com/stream.mp4",
                "rtsp://example.com/stream"
            ])
            url_layout.addWidget(self.url_input)
            self.input_layout.addLayout(url_layout)
            
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Media File", "", 
            "Media Files (*.mp4 *.avi *.mov *.mkv *.jpg *.png);;All Files (*)"
        )
        if file_path:
            self.file_input.setCurrentText(file_path)
            
    def on_conf_threshold_changed(self, value):
        self.conf_threshold = value / 100.0
        self.conf_value_label.setText(f"{self.conf_threshold:.2f}")
        
    def on_skip_frames_changed(self, value):
        self.skip_frames = value
        
    def on_img_size_changed(self, value):
        self.img_size = value
        
    def on_camera_index_changed(self, value):
        self.camera_index = value
        
    def switch_to_page(self, index):
        self.stacked_widget.setCurrentIndex(index)
        
    def load_model(self):
        try:
            self.model = YOLO('yolov8n.pt')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {e}")
            
    def start_detection(self):
        self.is_detecting = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Get input source
        if self.source_camera_radio.isChecked():
            self.current_source = self.camera_index_spinbox.value() if hasattr(self, 'camera_index_spinbox') else self.camera_index
        elif self.source_file_radio.isChecked():
            self.current_source = self.file_input.currentText() if hasattr(self, 'file_input') else ""
        elif self.source_url_radio.isChecked():
            self.current_source = self.url_input.currentText() if hasattr(self, 'url_input') else ""
            
        # Initialize camera or video source
        try:
            self.cap = cv2.VideoCapture(self.current_source)
            if not self.cap.isOpened():
                raise Exception("Cannot open video source")
                
            self._frame_idx = 0
            
            # Start timer
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # ~33 FPS
            
        except Exception as e:
            print(f"Cannot start detection: {str(e)}")
            self.stop_detection()
            
    def update_frame(self):
        if not self.is_detecting or not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            print("Cannot read frame")
            self.stop_detection()
            return
            
        self._frame_idx += 1
        
        # Skip frames
        if self._frame_idx % self.skip_frames != 0:
            return
            
        # Process frame
        processed_frame = self.detect_persons_in_frame(frame)
        
        # Display image
        self.display_frame(processed_frame)
        
    def detect_persons_in_frame(self, frame):
        if self.model is None:
            return frame
            
        try:
            # Resize image for faster processing
            h, w = frame.shape[:2]
            imgsz = self.img_size
            small = cv2.resize(frame, (imgsz, imgsz))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(small_rgb, imgsz=imgsz, conf=self.conf_threshold, classes=[0])
            
            person_count = 0
            if len(results) > 0:
                r = results[0]
                boxes = getattr(r, 'boxes', None)
                if boxes is not None and len(boxes) > 0:
                    # Scale coordinates to original frame size
                    scale_x = w / imgsz
                    scale_y = h / imgsz
                    for box in boxes:
                        # Get bounding box coordinates
                        xy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else np.array(box.xyxy[0])
                        x1, y1, x2, y2 = xy * np.array([scale_x, scale_y, scale_x, scale_y])
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                        person_count += 1
                        
                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'Person {conf:.2f}'
                        cv2.putText(frame, label, (x1, max(15, y1 - 10)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display person count
            cv2.putText(frame, f'Persons: {person_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                       
        except Exception as e:
            print(f"Detection error: {e}")
            
        return frame
        
    def display_frame(self, frame):
        # Convert color format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit display area
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.width(), 
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Update display in UI thread
        self.video_label.setPixmap(scaled_pixmap)
        
    def stop_detection(self):
        self.is_detecting = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        if hasattr(self, 'timer'):
            self.timer.stop()
            delattr(self, 'timer')
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        if hasattr(self, '_frame_idx'):
            delattr(self, '_frame_idx')
            
        # Clear video display
        self.video_label.setText("Video Display Area")
        print("Detection has been stopped")


def main():
    app = QApplication(sys.argv)
    # Set theme
    setTheme(Theme.AUTO)
    
    # Create and show main window
    window = PersonDetectionGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()