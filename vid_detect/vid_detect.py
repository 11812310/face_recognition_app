from ultralytics import YOLO
import torch
import cv2

model = YOLO("model.pt")

video_path = "video.mp4"

capture = cv2.VideoCapture(video_path)

if (capture.isOpened() == False):
    print("Video hasn't opened properly")

def mark_boxes(results_boxes, fr):
    for box in results_boxes:
        top_left_corner = (int(box.xyxy.tolist()[0][0]), int(box.xyxy.tolist()[0][1]))
        bottom_right_corner = (int(box.xyxy.tolist()[0][2]), int(box.xyxy.tolist()[0][3]))
        blue = (255, 0, 0)
        thickness = 2

        cv2.rectangle(fr, top_left_corner, bottom_right_corner, blue, thickness)
    return fr

marked_frames = []
i = 0

# for each frame
while(capture.isOpened()):

    retval, frame = capture.read()

    if retval:
        # detect faces
        results_frame = model(frame)
        # mark faces with boxes
        marked_frame = mark_boxes(results_frame[0].boxes, frame)
        marked_frames.append(marked_frame)

    else:
        break

    # testing condition, just for 1000 first frames   
    if i > 1000:
        break

    i += 1
        
fps = capture.get(cv2.CAP_PROP_FPS)        
processed_video = cv2.VideoWriter('processed_video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (1280,720))

for f in marked_frames:
  processed_video.write(f)

processed_video.release()

        







