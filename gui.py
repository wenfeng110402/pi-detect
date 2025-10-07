import sys
import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QFileDialog, QStackedWidget, QButtonGroup, QRadioButton, QMessageBox, QLineEdit, QCheckBox)
from qfluentwidgets import NavigationInterface, NavigationItemPosition, FluentWindow
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (SubtitleLabel, PrimaryPushButton, Slider, SpinBox, ComboBox,
                           setTheme, Theme, InfoBar, InfoBarPosition, PushButton, LineEdit, CardWidget, SwitchButton)
from ultralytics import YOLO
import numpy as np
from pathlib import Path


class PersonDetectionGUI(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pi-Detect")
        self.resize(1000, 700)
        
        # Initialize variables
        self.model = None
        self.pose_model = None
        self.cap = None
        self.is_detecting = False
        self.is_image_mode = False
        self.current_source = None
        self.conf_threshold = 0.3
        self.skip_frames = 2
        self.img_size = 416
        self.enable_pose_detection = False
        
        # Initialize source variables
        self.camera_index = 0
        self.file_path = ""
        self.url = ""
        
        # Model variables - 只保留YOLOv11模型
        self.available_models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        
        # Pose detection models - 只保留YOLOv11的pose模型
        self.pose_models = ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt"]
        
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
        
        '''
        # Title
        title = SubtitleLabel("Pi Dectect")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        '''
        # Media source selection
        source_card = CardWidget()
        source_card.setBorderRadius(8)
        source_layout = QVBoxLayout(source_card)
        
        source_title = SubtitleLabel("Media Source")
        source_layout.addWidget(source_title)
        
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
        self.camera_spinbox = SpinBox()
        self.camera_spinbox.setRange(0, 10)
        self.camera_spinbox.setValue(0)
        camera_layout.addWidget(self.camera_spinbox)
        self.input_layout.addLayout(camera_layout)
        
        source_layout.addLayout(self.input_layout)
        layout.addWidget(source_card)
        
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
        conf_card = CardWidget()
        conf_card.setBorderRadius(8)
        conf_layout = QHBoxLayout(conf_card)
        conf_layout.addWidget(QLabel("Threshold:"))
        self.conf_slider = Slider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(30)
        self.conf_value_label = QLabel("0.30")
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        layout.addWidget(conf_card)
        
        # Skip frames setting
        skip_card = CardWidget()
        skip_card.setBorderRadius(8)
        skip_layout = QHBoxLayout(skip_card)
        skip_layout.addWidget(QLabel("Skip Frames:"))
        self.skip_spinbox = SpinBox()
        self.skip_spinbox.setRange(1, 10)
        self.skip_spinbox.setValue(2)
        skip_layout.addWidget(self.skip_spinbox)
        layout.addWidget(skip_card)
        
        # Image size setting
        size_card = CardWidget()
        size_card.setBorderRadius(8)
        size_layout = QHBoxLayout(size_card)
        size_layout.addWidget(QLabel("Size:"))
        self.size_spinbox = SpinBox()
        self.size_spinbox.setRange(320, 1024)
        self.size_spinbox.setValue(416)
        self.size_spinbox.setSingleStep(32)
        size_layout.addWidget(self.size_spinbox)
        layout.addWidget(size_card)
        
        # Camera setting
        camera_card = CardWidget()
        camera_card.setBorderRadius(8)
        camera_layout = QHBoxLayout(camera_card)
        camera_layout.addWidget(QLabel("Default Camera Index:"))
        self.camera_index_spinbox = SpinBox()
        self.camera_index_spinbox.setRange(0, 10)
        self.camera_index_spinbox.setValue(0)
        camera_layout.addWidget(self.camera_index_spinbox)
        layout.addWidget(camera_card)
        
        # Model selection setting
        model_card = CardWidget()
        model_card.setBorderRadius(8)
        model_layout = QVBoxLayout(model_card)
        
        # Model selection combo
        model_selection_layout = QHBoxLayout()
        model_selection_layout.addWidget(QLabel("Model:"))
        self.model_combo = ComboBox()
        # Populate with available models directly without update_model_combo method
        for model in self.available_models:
            self.model_combo.addItem(model)
        model_selection_layout.addWidget(self.model_combo)
        model_layout.addLayout(model_selection_layout)
        
        layout.addWidget(model_card)
        
        # Pose detection setting
        pose_card = CardWidget()
        pose_card.setBorderRadius(8)
        pose_layout = QVBoxLayout(pose_card)
        
        # Pose detection enable switch
        pose_enable_layout = QHBoxLayout()
        pose_enable_layout.addWidget(QLabel("Enable Pose Detection:"))
        self.pose_enable_switch = SwitchButton()
        self.pose_enable_switch.checkedChanged.connect(self.on_pose_enable_changed)
        pose_enable_layout.addWidget(self.pose_enable_switch)
        pose_layout.addLayout(pose_enable_layout)
        
        # Pose model selection combo
        pose_model_layout = QHBoxLayout()
        pose_model_layout.addWidget(QLabel("Pose Model:"))
        self.pose_model_combo = ComboBox()
        # Populate with available pose models directly without update_pose_model_combo method
        for model in self.pose_models:
            self.pose_model_combo.addItem(model)
        self.pose_model_combo.setEnabled(False)  # Disabled by default
        pose_model_layout.addWidget(self.pose_model_combo)
        pose_layout.addLayout(pose_model_layout)
        
        layout.addWidget(pose_card)
        
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
        
        description_card = CardWidget()
        description_card.setBorderRadius(8)
        description_layout = QVBoxLayout(description_card)
        
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
        description_layout.addWidget(description)
        
        layout.addWidget(description_card)
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
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.pose_model_combo.currentTextChanged.connect(self.on_pose_model_changed)
        self.pose_enable_switch.checkedChanged.connect(self.on_pose_enable_changed)
        
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
            self.camera_index_spinbox = SpinBox()
            self.camera_index_spinbox.setRange(0, 10)
            self.camera_index_spinbox.setValue(self.camera_index)
            camera_layout.addWidget(self.camera_index_spinbox)
            self.input_layout.addLayout(camera_layout)
            
        elif self.source_file_radio.isChecked():
            # File input
            file_layout = QHBoxLayout()
            file_layout.addWidget(QLabel("File Path:"))
            self.file_input = LineEdit()  # 使用LineEdit替代ComboBox
            self.file_input.setText("example.mp4")
            self.file_browse_button = PushButton("Browse...")
            self.file_browse_button.clicked.connect(self.browse_file)
            file_layout.addWidget(self.file_input)
            file_layout.addWidget(self.file_browse_button)
            self.input_layout.addLayout(file_layout)
            
        elif self.source_url_radio.isChecked():
            # URL input
            url_layout = QHBoxLayout()
            url_layout.addWidget(QLabel("URL:"))
            self.url_input = LineEdit()
            self.url_input.setText("http://example.com/stream.mp4")
            url_layout.addWidget(self.url_input)
            self.input_layout.addLayout(url_layout)
            
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Media File", "", 
            "Media Files (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp *.tiff);;All Files (*)"
        )
        if file_path:
            self.file_input.setText(file_path)
            
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
            model_name = self.model_combo.currentText() if hasattr(self, 'model_combo') and self.model_combo.currentText() else 'yolo11n.pt'
            self.model = YOLO(model_name)
            print(f"Model {model_name} loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
        
    def load_pose_model(self):
        """加载姿态检测模型"""
        try:
            if not self.enable_pose_detection:
                return
                
            pose_model_name = self.pose_model_combo.currentText() if hasattr(self, 'pose_model_combo') and self.pose_model_combo.currentText() else 'yolo11n-pose.pt'
            self.pose_model = YOLO(pose_model_name)
            print(f"Pose model {pose_model_name} loaded successfully")
        except Exception as e:
            print(f"Pose model loading failed: {str(e)}")
        
    def on_model_changed(self, model_text):
        """当模型选择改变时重新加载模型"""
        self.load_model()
        
    def on_pose_model_changed(self, model_text):
        """当姿态模型选择改变时重新加载模型"""
        if self.enable_pose_detection:
            self.load_pose_model()
            
    def on_pose_enable_changed(self, checked):
        """当姿态检测启用状态改变时"""
        self.enable_pose_detection = checked
        self.pose_model_combo.setEnabled(checked)
        if checked:
            self.load_pose_model()
            
    def start_detection(self):
        self.is_detecting = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Get input source
        if self.source_camera_radio.isChecked():
            self.current_source = self.camera_index_spinbox.value() if hasattr(self, 'camera_index_spinbox') else self.camera_index
            self.is_image_mode = False
        elif self.source_file_radio.isChecked():
            self.current_source = self.file_input.text() if hasattr(self, 'file_input') else ""
            # Check if it's an image file
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            file_ext = Path(self.current_source).suffix.lower()
            self.is_image_mode = file_ext in image_extensions
        elif self.source_url_radio.isChecked():
            self.current_source = self.url_input.text() if hasattr(self, 'url_input') else ""
            self.is_image_mode = False
            
        # Handle image mode differently from video mode
        if self.is_image_mode:
            self.process_image_file()
        else:
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
            
    def process_image_file(self):
        """处理图像文件"""
        try:
            # Read image file
            frame = cv2.imread(self.current_source)
            if frame is None:
                raise Exception("Cannot read image file")
                
            # Process the image
            processed_frame = self.detect_persons_in_frame(frame)
            
            # Display the result
            self.display_frame(processed_frame)
            
            # Stop detection since it's a single image
            self.is_detecting = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
        except Exception as e:
            print(f"Cannot process image file: {str(e)}")
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
            
            # Run pose detection if enabled
            if self.enable_pose_detection and self.pose_model is not None:
                pose_results = self.pose_model(small_rgb, imgsz=imgsz, conf=self.conf_threshold)
                if len(pose_results) > 0:
                    pose_r = pose_results[0]
                    keypoints = getattr(pose_r, 'keypoints', None)
                    if keypoints is not None:
                        # Scale keypoints to original frame size
                        for kp in keypoints.xy:
                            if len(kp) > 0:
                                scaled_kp = kp.cpu().numpy() * np.array([scale_x, scale_y])
                                for point in scaled_kp.astype(int):
                                    cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
            
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