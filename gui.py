import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
from ultralytics import YOLO
import numpy as np
import requests
from PIL import Image, ImageTk
import os
import time
from urllib.parse import urlparse


class PersonDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pi-Detect 人体检测系统")
        self.root.geometry("800x600")

        # 初始化变量
        self.model = None
        self.cap = None
        self.is_detecting = False
        self.is_image_mode = False  # 标记是否为图片模式
        self.current_source = None
        self.conf_threshold = 0.3
        self.skip_frames = 2
        self.img_size = 416
        
        # 初始化文件路径变量
        self.camera_index_var = tk.StringVar(value="0")
        self.file_path_var = tk.StringVar()
        self.url_var = tk.StringVar()

        # 创建界面
        self.create_widgets()

        # 加载模型
        self.load_model()

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text="人体检测系统", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # 媒体源选择框架
        source_frame = ttk.LabelFrame(main_frame, text="选择媒体源", padding="10")
        source_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        source_frame.columnconfigure(1, weight=1)

        # 源类型选择
        self.source_var = tk.StringVar(value="camera")
        camera_radio = ttk.Radiobutton(source_frame, text="摄像头", variable=self.source_var, value="camera",
                                       command=self.on_source_change)
        file_radio = ttk.Radiobutton(source_frame, text="本地文件", variable=self.source_var, value="file",
                                     command=self.on_source_change)
        url_radio = ttk.Radiobutton(source_frame, text="网络链接", variable=self.source_var, value="url",
                                    command=self.on_source_change)

        camera_radio.grid(row=0, column=0, padx=(0, 20))
        file_radio.grid(row=0, column=1, padx=(0, 20))
        url_radio.grid(row=0, column=2, padx=(0, 20))

        # 源输入框架
        self.input_frame = ttk.Frame(source_frame)
        self.input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        self.input_frame.columnconfigure(1, weight=1)

        # 初始化输入组件
        self.init_source_inputs()

        # 参数设置框架
        param_frame = ttk.LabelFrame(main_frame, text="检测参数", padding="10")
        param_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        # 置信度阈值
        ttk.Label(param_frame, text="置信度阈值:").grid(row=0, column=0, sticky=tk.W)
        self.conf_var = tk.DoubleVar(value=self.conf_threshold)
        conf_scale = ttk.Scale(param_frame, from_=0.01, to=1.0, variable=self.conf_var, orient=tk.HORIZONTAL)
        conf_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        self.conf_label = ttk.Label(param_frame, text=f"{self.conf_var.get():.2f}")
        self.conf_label.grid(row=0, column=2, sticky=tk.W, padx=(10, 0))

        # 跳帧数
        ttk.Label(param_frame, text="跳帧数:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.skip_var = tk.IntVar(value=self.skip_frames)
        skip_scale = ttk.Scale(param_frame, from_=1, to=10, variable=self.skip_var, orient=tk.HORIZONTAL)
        skip_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(10, 0))
        self.skip_label = ttk.Label(param_frame, text=str(self.skip_var.get()))
        self.skip_label.grid(row=1, column=2, sticky=tk.W, padx=(10, 0), pady=(10, 0))

        # 图像尺寸
        ttk.Label(param_frame, text="图像尺寸:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.size_var = tk.IntVar(value=self.img_size)
        size_scale = ttk.Scale(param_frame, from_=224, to=640, variable=self.size_var, orient=tk.HORIZONTAL)
        size_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(10, 0))
        self.size_label = ttk.Label(param_frame, text=str(self.size_var.get()))
        self.size_label.grid(row=2, column=2, sticky=tk.W, padx=(10, 0), pady=(10, 0))

        # 绑定参数变化事件
        conf_scale.configure(command=self.on_conf_change)
        skip_scale.configure(command=self.on_skip_change)
        size_scale.configure(command=self.on_size_change)

        # 控制按钮框架
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10))

        self.start_button = ttk.Button(control_frame, text="开始检测", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_button = ttk.Button(control_frame, text="停止检测", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))

        # 视频显示区域
        display_frame = ttk.LabelFrame(main_frame, text="视频显示", padding="10")
        display_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        self.video_label = ttk.Label(display_frame)
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

    def init_source_inputs(self):
        # 清除之前的输入组件
        for widget in self.input_frame.winfo_children():
            widget.destroy()

        source_type = self.source_var.get()

        if source_type == "camera":
            ttk.Label(self.input_frame, text="摄像头索引:").grid(row=0, column=0, sticky=tk.W)
            camera_entry = ttk.Entry(self.input_frame, textvariable=self.camera_index_var)
            camera_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))

        elif source_type == "file":
            ttk.Label(self.input_frame, text="文件路径:").grid(row=0, column=0, sticky=tk.W)
            file_entry = ttk.Entry(self.input_frame, textvariable=self.file_path_var)
            file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
            file_button = ttk.Button(self.input_frame, text="浏览", command=self.browse_file)
            file_button.grid(row=0, column=2, padx=(10, 0))

        elif source_type == "url":
            ttk.Label(self.input_frame, text="网络链接:").grid(row=0, column=0, sticky=tk.W)
            url_entry = ttk.Entry(self.input_frame, textvariable=self.url_var)
            url_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
            download_button = ttk.Button(self.input_frame, text="下载并使用", command=self.download_and_use)
            download_button.grid(row=0, column=2, padx=(10, 0))

    def on_source_change(self):
        self.init_source_inputs()

    def on_conf_change(self, value):
        self.conf_threshold = float(value)
        self.conf_label.config(text=f"{self.conf_threshold:.2f}")

    def on_skip_change(self, value):
        self.skip_frames = int(float(value))
        self.skip_label.config(text=str(self.skip_frames))

    def on_size_change(self, value):
        self.img_size = int(float(value))
        self.size_label.config(text=str(self.img_size))

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="选择媒体文件",
            filetypes=[
                ("媒体文件", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp *.tiff"),
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def get_filename_from_url(self, url):
        # 从URL中提取文件名
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path)
        
        # 如果没有扩展名，根据内容类型添加
        if not os.path.splitext(file_name)[1]:
            try:
                response = requests.head(url, timeout=10)
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type:
                    if 'jpeg' in content_type or 'jpg' in content_type:
                        file_name += '.jpg'
                    elif 'png' in content_type:
                        file_name += '.png'
                    else:
                        file_name += '.jpg'  # 默认
                else:
                    file_name += '.mp4'  # 默认视频
            except:
                file_name += '.mp4'  # 默认
                
        return file_name

    def download_and_use(self):
        url = self.url_var.get()
        if not url:
            messagebox.showerror("错误", "请输入有效的网络链接")
            return

        try:
            self.status_var.set("正在下载文件...")
            self.root.update()

            # 设置请求头，模拟浏览器访问
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # 获取文件名
            filename = self.get_filename_from_url(url)
            
            # 确保文件名唯一
            counter = 1
            original_filename = filename
            while os.path.exists(filename):
                name, ext = os.path.splitext(original_filename)
                filename = f"{name}_{counter}{ext}"
                counter += 1

            # 保存文件
            with open(filename, "wb") as f:
                f.write(response.content)

            self.file_path_var.set(filename)
            self.status_var.set(f"文件下载完成: {filename}")
            messagebox.showinfo("成功", f"文件下载完成: {filename}")

        except requests.exceptions.RequestException as e:
            self.status_var.set("网络请求错误")
            messagebox.showerror("网络错误", f"下载文件时网络出错: {str(e)}")
        except Exception as e:
            self.status_var.set("下载失败")
            messagebox.showerror("错误", f"下载文件时出错: {str(e)}")

    def load_model(self):
        try:
            self.status_var.set("正在加载模型...")
            self.root.update()
            self.model = YOLO('yolov8n.pt')
            self.status_var.set("模型加载完成，准备就绪")
        except Exception as e:
            self.status_var.set("模型加载失败")
            messagebox.showerror("错误", f"加载模型时出错: {str(e)}")

    def start_detection(self):
        if self.model is None:
            messagebox.showerror("错误", "模型未加载，请稍后重试")
            return

        source_type = self.source_var.get()

        try:
            # 检查是否为图片文件
            self.is_image_mode = False
            if source_type in ["file", "url"]:
                file_path = self.file_path_var.get()
                if file_path:
                    # 获取文件扩展名
                    ext = os.path.splitext(file_path)[1].lower()
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                    if ext in image_extensions:
                        self.is_image_mode = True

            if self.is_image_mode:
                # 图片模式处理
                file_path = self.file_path_var.get()
                if not file_path or not os.path.exists(file_path):
                    messagebox.showerror("错误", "请选择有效的文件路径")
                    return
                    
                self.is_detecting = True
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.status_var.set("正在处理图片...")

                # 在新线程中运行图片处理
                self.detection_thread = threading.Thread(target=self.process_image, args=(file_path,), daemon=True)
                self.detection_thread.start()
            else:
                # 视频/摄像头模式处理
                if source_type == "camera":
                    camera_index = int(self.camera_index_var.get())
                    self.cap = cv2.VideoCapture(camera_index)
                elif source_type == "file":
                    file_path = self.file_path_var.get()
                    if not file_path or not os.path.exists(file_path):
                        messagebox.showerror("错误", "请选择有效的文件路径")
                        return
                    self.cap = cv2.VideoCapture(file_path)
                elif source_type == "url":
                    # 使用已下载的文件
                    file_path = self.file_path_var.get()
                    if not file_path or not os.path.exists(file_path):
                        messagebox.showerror("错误", "请先下载文件")
                        return
                    self.cap = cv2.VideoCapture(file_path)

                if not self.cap.isOpened():
                    messagebox.showerror("错误", "无法打开媒体源")
                    return

                self.is_detecting = True
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.status_var.set("正在检测...")

                # 在新线程中运行检测
                self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
                self.detection_thread.start()

        except Exception as e:
            self.status_var.set("启动检测失败")
            messagebox.showerror("错误", f"启动检测时出错: {str(e)}")

    def process_image(self, image_path):
        try:
            # 读取图片
            frame = cv2.imread(image_path)
            if frame is None:
                self.status_var.set("无法读取图片文件")
                return

            # 处理图片
            processed_frame = self.detect_persons_in_frame(frame)
            
            # 显示结果
            self.display_frame(processed_frame)
            self.status_var.set("图片处理完成")
            
        except Exception as e:
            self.status_var.set(f"处理图片时出错: {str(e)}")
        finally:
            # 图片模式下，保持检测状态为False，但允许重新开始
            self.is_detecting = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def detect_persons_in_frame(self, frame):
        # 获取原始帧尺寸
        h, w = frame.shape[:2]
        imgsz = self.img_size

        # 调整图像大小
        small = cv2.resize(frame, (imgsz, imgsz))
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # 运行推理
        results = self.model(small_rgb, imgsz=imgsz, conf=self.conf_threshold, classes=[0])

        person_count = 0
        if len(results) > 0:
            r = results[0]
            boxes = getattr(r, 'boxes', None)
            if boxes is not None and len(boxes) > 0:
                # 计算缩放比例
                scale_x = w / imgsz
                scale_y = h / imgsz
                for box in boxes:
                    # 获取边界框坐标
                    xy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else np.array(box.xyxy[0])
                    x1, y1, x2, y2 = xy * np.array([scale_x, scale_y, scale_x, scale_y])
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    person_count += 1

                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'Person {conf:.2f}'
                    cv2.putText(frame, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 绘制人数
        cv2.putText(frame, f'Persons: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame

    def run_detection(self):
        frame_idx = 0
        last_time = time.time()
        processed = 0

        while self.is_detecting and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # 视频结束，自动停止
                break

            frame_idx += 1

            # 跳帧处理
            if (frame_idx % self.skip_frames) != 0:
                continue

            # 处理帧
            processed_frame = self.detect_persons_in_frame(frame)
            
            # 计算FPS
            processed += 1
            now = time.time()
            fps = processed / (now - last_time) if now > last_time else 0.0
            
            # 添加FPS信息
            cv2.putText(processed_frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 显示图像
            self.display_frame(processed_frame)

        # 释放资源
        if self.cap:
            self.cap.release()
            
        # 更新UI状态
        self.is_detecting = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        if self.status_var.get() == "正在检测...":
            self.status_var.set("检测完成")

    def display_frame(self, frame):
        # 转换颜色格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转换为PIL图像
        pil_image = Image.fromarray(frame_rgb)
        # 调整大小以适应显示区域
        pil_image.thumbnail((750, 400))
        # 转换为Tkinter图像
        tk_image = ImageTk.PhotoImage(pil_image)

        # 在主线程中更新UI
        self.root.after(0, self.update_video_label, tk_image)

    def update_video_label(self, image):
        self.video_label.configure(image=image)
        self.video_label.image = image  # 保持引用

    def stop_detection(self):
        self.is_detecting = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("检测已停止")


def main():
    root = tk.Tk()
    app = PersonDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()