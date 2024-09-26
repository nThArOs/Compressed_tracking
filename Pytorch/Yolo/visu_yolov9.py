import cv2
from ultralytics import YOLO

#Load Yolov9
#model = YOLO('yolov8-residual-last.pt')
model = YOLO('yolov8n.pt')
# open the video file
video_path = r"full.mp4"
cap = cv2.VideoCapture(video_path)

codec = cv2.VideoWriter_fourcc(*"MJPG")

out = cv2.VideoWriter('result/full.avi', codec, 30, (1920, 1080))

while cap.isOpened():
    #Read a frame from the video
    success, frame = cap.read()

    if success:
        #run the YLOv9 tracking
        conf = 0.5
        iou = 0.5
        results = model.track(frame, persist=True, conf=conf,iou=iou, show=True, tracker="bytetrack.yaml")

        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        cv2.imshow("Yolov9 Tracking", annotated_frame)