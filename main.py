import cv2
from ultralytics import YOLO

# 加载预训练的YOLOv8n模型
model = YOLO('yolov8n.pt')

def main():
    # 初始化树莓派摄像头
    cap = cv2.VideoCapture(0)
    
    # 设置摄像头参数（根据需要调整分辨率）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    while True:
        # 从摄像头读取帧
        ret, frame = cap.read()
        
        if not ret:
            print("无法获取画面")
            break
        
        # 使用YOLOv8模型进行推理
        results = model(frame)
        
        # 初始化人物计数
        person_count = 0
        
        # 处理检测结果
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取类别ID
                    cls_id = int(box.cls[0])
                    
                    # 只处理人物类别 (通常类别ID为0)
                    if cls_id == 0:
                        # 增加人物计数
                        person_count += 1
                        
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # 在图像上绘制边界框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 添加标签
                        label = f'Person {box.conf[0]:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 在图像上显示人物数量
        cv2.putText(frame, f'Persons: {person_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 显示结果帧
        cv2.imshow('Person Detection and Counting', frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()