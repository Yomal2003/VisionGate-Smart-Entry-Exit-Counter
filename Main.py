# Import required libraries
import cv2 
from ultralytics import YOLO
import cvzone
import os

# === Load YOLOv8 model ===
model_path = 'yolo12n.pt'
if not os.path.exists(model_path):
    print(f"Model file '{model_path}' not found!")
    exit()
model = YOLO(model_path)
names = model.names

# === Define vertical line's X position ===
line_x = 443

# === Track previous center positions ===
hist = {}

# === IN/OUT counters ===
in_count = 0
out_count = 0

# === Open video file or webcam ===
video_path = "C:/Users/Yomal/Downloads/People_Counting/video/video.mp4"
if not os.path.exists(video_path):
    print(f"Video file not found at '{video_path}'. Trying webcam...")
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to open video or webcam.")
    exit()

# === Define the mouse callback function ===
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

# === Create a named OpenCV window and set the mouse callback ===
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended or failed to read frame.")
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))

    # === Detect and track persons (class 0) ===
    results = model.track(frame, persist=True, classes=[0])

    if results and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = box
            c = names[class_id]
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)

            # Draw bounding box and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{c.upper()}', (x1, y1 - 10), scale=1, thickness=1,
                               colorT=(255, 255, 255), colorR=(0, 0, 255), offset=5, border=2)
            cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y2 + 10), scale=1, thickness=2,
                               colorT=(255, 255, 255), colorR=(0, 255, 0), offset=5, border=2)

            # IN/OUT Logic
            if track_id in hist:
                prev_cx, prev_cy = hist[track_id]
                if prev_cx < line_x <= cx:
                    in_count += 1
                if prev_cx > line_x >= cx:
                    out_count += 1
            hist[track_id] = (cx, cy)

    # === Display counts using cvzone ===
    cvzone.putTextRect(frame, f'IN: {in_count}', (40, 60), scale=2, thickness=2,
                       colorT=(255, 255, 255), colorR=(0, 128, 0))
    cvzone.putTextRect(frame, f'OUT: {out_count}', (40, 100), scale=2, thickness=2,
                       colorT=(255, 255, 255), colorR=(0, 0, 255))

    # === Draw vertical line ===
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 1)

    # === Show the frame ===
    cv2.imshow("RGB", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        print("Exiting on ESC key.")
        break

# === Release and close ===
cap.release()
cv2.destroyAllWindows()

# === Print final results in terminal ===
print("\n========== FINAL RESULTS ==========")
print(f"Total people entered  ➜ IN  : {in_count}")
print(f"Total people exited   ➜ OUT : {out_count}")
print("===================================")
