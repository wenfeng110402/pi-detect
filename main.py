import argparse
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO


class ThreadedCamera:
    """Background thread for camera capture to avoid blocking on read()."""
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            if not ret:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser(description="Lightweight YOLO person detection optimized for Raspberry Pi")
    p.add_argument('--source', type=int, default=0, help='camera source (default 0)')
    p.add_argument('--imgsz', type=int, default=416, help='inference image size (square)')
    p.add_argument('--skip', type=int, default=2, help='process every Nth frame (higher => fewer inferences)')
    p.add_argument('--conf', type=float, default=0.3, help='confidence threshold')
    p.add_argument('--show', action='store_true', help='show display window')
    p.add_argument('--device', type=str, default='cpu', help='inference device, e.g. cpu or 0 for cuda')
    return p.parse_args()


def main():
    args = parse_args()

    # Load model once
    model = YOLO('yolov8n.pt')

    cam = ThreadedCamera(src=args.source, width=640, height=480)
    time.sleep(0.2)  # allow camera thread to warm up

    if not cam.ret:
        print('Camera not available')
        return

    last_time = time.time()
    processed = 0
    frame_idx = 0
    display_frame = None

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print('No frame, exiting')
                break

            frame_idx += 1

            # Only run inference on every `skip` frames to save CPU
            if (frame_idx % args.skip) != 0:
                # show last processed frame if required
                if args.show and display_frame is not None:
                    cv2.imshow('Person Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # Prepare small square image for faster inference
            h, w = frame.shape[:2]
            imgsz = args.imgsz
            small = cv2.resize(frame, (imgsz, imgsz))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # Run inference; filter to class 0 (person) and confidence threshold
            results = model(small_rgb, imgsz=imgsz, conf=args.conf, classes=[0])

            person_count = 0
            if len(results) > 0:
                r = results[0]
                boxes = getattr(r, 'boxes', None)
                if boxes is not None and len(boxes) > 0:
                    # boxes.xyxy are in resized image coordinates -> scale to original frame
                    scale_x = w / imgsz
                    scale_y = h / imgsz
                    for box in boxes:
                        # each box.xyxy is [x1,y1,x2,y2]
                        xy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else np.array(box.xyxy[0])
                        x1, y1, x2, y2 = xy * np.array([scale_x, scale_y, scale_x, scale_y])
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                        conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                        person_count += 1

                        # Draw boxes and labels onto a copy of original frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'Person {conf:.2f}'
                        cv2.putText(frame, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Overlay count and FPS
            processed += 1
            now = time.time()
            fps = processed / (now - last_time) if now > last_time else 0.0
            cv2.putText(frame, f'Persons: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            display_frame = frame
            if args.show:
                cv2.imshow('Person Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()