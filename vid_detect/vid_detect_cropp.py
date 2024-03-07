from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
from pathlib import Path
import torch
import cv2

model = YOLO("model.pt")

VIDEO_PATH = "torreto.mp4"

capture = cv2.VideoCapture(VIDEO_PATH)

if (capture.isOpened() == False):
    print("Video hasn't opened properly")

def mark_box(box, fr):
    top_left_corner = (int(box.xyxy.tolist()[0][0]), int(box.xyxy.tolist()[0][1]))
    bottom_right_corner = (int(box.xyxy.tolist()[0][2]), int(box.xyxy.tolist()[0][3]))
    blue = (255, 0, 0)
    thickness = 2
    cv2.rectangle(fr, top_left_corner, bottom_right_corner, blue, thickness)
    return fr

marked_frames = []
i = 0

# for each frame:
while(capture.isOpened()):

    retval, frame = capture.read()

    if retval:
        frame_cpy = frame.copy()
        # detect faces
        results_frame = model(frame)
        j = 0
        for box in results_frame[0].boxes:
            save_one_box(box.xyxy, frame, file= Path(f'./temp/frame{i}_face{j}.jpg'), BGR=True)
            j += 1
            marked_frame = mark_box(box, frame_cpy)
            marked_frames.append(marked_frame)
    else:
        break

    # testing condition, just for 200 first frames   
    if i > 10:
        break

    i += 1

fps = capture.get(cv2.CAP_PROP_FPS)        
processed_video = cv2.VideoWriter('processed_torreto_cropping.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (1280,720))

for f in marked_frames:
  processed_video.write(f)

processed_video.release()
        







