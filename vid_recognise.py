from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
from pathlib import Path
import torch
import cv2
from deepface import DeepFace

model = YOLO("model.pt")

VIDEO_PATH = "vin_diesel_video.mp4"

capture = cv2.VideoCapture(VIDEO_PATH)

if (capture.isOpened() == False):
    print("Video hasn't opened properly")

def mark_box(box, fr):
    top_left_corner = (int(box.xyxy.tolist()[0][0]), int(box.xyxy.tolist()[0][1]))
    bottom_right_corner = (int(box.xyxy.tolist()[0][2]), int(box.xyxy.tolist()[0][3]))
    red = (0, 0, 255)
    thickness = 2
    cv2.rectangle(fr, top_left_corner, bottom_right_corner, red, thickness)
    return fr
        
def recognise_face(image_path, target_name): 
    dfs = DeepFace.find(img_path=image_path, db_path="database", enforce_detection=False) # possibly delete last param
    if dfs and len(dfs[0]['identity']) > 0:
        name = dfs[0]['identity'].to_string(index=False).split('\\')[1]
        print(name)
        if name == target_name:
            return True
        else:
            return False

# TODO: create logfile for given video and name

TARGET_NAME = "Vin_Diesel"
processed_frames = []
i = 0

# for each frame:
while(capture.isOpened()):

    retval, frame = capture.read()

    if retval:
        frame_cpy = frame.copy()
        # detect faces
        results_frame = model(frame)
        # for each box = detected face:
        j = 0
        for box in results_frame[0].boxes:
            # save cropped face as image
            image_path = Path(f'./temp/frame{i}_face{j}.jpg')
            save_one_box(box.xyxy, frame, file=Path(f'./temp/frame{i}_face{j}.jpg'), BGR=True)
            j += 1
            if recognise_face(image_path, TARGET_NAME) == True:
                marked_frame = mark_box(box, frame_cpy)
                processed_frames.append(marked_frame)
                # TODO: update logfile with a new line
            else:
                processed_frames.append(frame)
            
    else:
        break
    # testing condition, just for n first frames   
    if i > 200:
        break

    i += 1
        
fps = capture.get(cv2.CAP_PROP_FPS)        
processed_video = cv2.VideoWriter('processed_video_long2.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (1280,720))

for f in processed_frames:
  processed_video.write(f)

processed_video.release()

        







