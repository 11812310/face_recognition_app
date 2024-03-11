from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
from pathlib import Path
import torch
import cv2
from deepface import DeepFace
from flask import Flask, request
import json
# from vid_recognise import vid_recognise

model = YOLO("model.pt")

class Query:
    def __init__(self, vid_name, persons):
        self.vid_name = vid_name
        self.persons = []
        for person in persons:
            self.persons.append(person)
            
class Answer:
    def __init__(self, output_vid_name, logfile_name):
        self.output_vid_name = output_vid_name
        self.logfile_name = logfile_name  
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)  

def vid_recognise(query: Query) -> Answer:
    
    vid_path = f"{query.vid_name}.mp4"

    capture = cv2.VideoCapture(vid_path)

    if (capture.isOpened() == False):
        return("Video hasn't opened properly")

    def mark_box(box, fr):
        top_left_corner = (int(box.xyxy.tolist()[0][0]), int(box.xyxy.tolist()[0][1]))
        bottom_right_corner = (int(box.xyxy.tolist()[0][2]), int(box.xyxy.tolist()[0][3]))
        red = (0, 0, 255)
        thickness = 2
        cv2.rectangle(fr, top_left_corner, bottom_right_corner, red, thickness)
        return fr
            
    def recognise_face(time_signature, image_path, target_names): 
        dfs = DeepFace.find(img_path=image_path, db_path="database", enforce_detection=False) # possibly choose different model
        if dfs and len(dfs[0]['identity']) > 0:
            name = dfs[0]['identity'].to_string(index=False).split('\\')[1]
            for target_name in target_names:
                if name == target_name:
                    print(name)
                    log.write(f"{time_signature}: {name} detected \n")
                    return True
            return False

    logfile_name = f"logfile_app_processed_{query.vid_name}"
    for person in query.persons:
        logfile_name += f"_{person}"
    logfile_name += ".txt"
    log = open(f"{logfile_name}.txt", "w") 

    processed_frames = []
    i = 0

    # for each frame:
    while(capture.isOpened()):

        retval, frame = capture.read()
        timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)

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
                # TODO: swap for a real time signature if that part will work 
                if recognise_face(timestamp, image_path, query.persons) == True:
                    marked_frame = mark_box(box, frame_cpy)
                    processed_frames.append(marked_frame)
                else:
                    processed_frames.append(frame)  
        else:
            break
        # testing condition, just for n first frames   
        if i > 10:
            break

        i += 1
            
    fps = capture.get(cv2.CAP_PROP_FPS)
    procesessed_video_name = f"processed_app_{query.vid_name}"
    for person in query.persons:
        procesessed_video_name += f"_{person}"
    procesessed_video_name += ".mp4"
    processed_video = cv2.VideoWriter(procesessed_video_name, cv2.VideoWriter_fourcc(*'MP4V'), fps, (1280,720))

    for f in processed_frames:
        processed_video.write(f)

    processed_video.release()

    log.close()

    answer = Answer(procesessed_video_name, logfile_name)

    return(answer)

recognition_app = Flask(__name__)

@recognition_app.route('/recognise', methods=['GET'])
def get_recognised(): 
    data = request.get_json()
    query = Query(data["vid_name"], data["persons"])
    recognised = vid_recognise(query)
    return recognised.toJSON()

@recognition_app.route("/health")
def healthcheck():
    return "Recognition app is up!"


            







