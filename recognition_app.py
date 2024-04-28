from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
from pathlib import Path
import torch
import cv2
from deepface import DeepFace
from flask import Flask, request
import json
from minio import Minio
import urllib3

model = YOLO("model.pt")

class Query:
    def __init__(self, vid_name, persons):
        self.vid_name = vid_name
        self.persons = []
        parsed_persons = json.loads(json.dumps(persons)) 
        print("PARSED PERSONS: ")
        print(parsed_persons)
        for person in parsed_persons:
            parsed_person = json.loads(json.dumps(person))
            print(parsed_person['person'])
            self.persons.append(parsed_person['person'])

class Answer:
    def __init__(self, output_vid_name, logfile_name):
        self.output_vid_name = output_vid_name
        self.logfile_name = logfile_name  
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4) 

def vid_recognise(query: Query, client) -> Answer:
    
    # download the video on which the recognition will run from minio and save it locally
    vid_path = f"{query.vid_name}.mp4"
    client.fget_object("original-videos-bucket", vid_path, vid_path)

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
            # TODO: check all detected names not just the first one detected 
            name = dfs[0]['identity'].to_string(index=False).split('/')[1]
            for target_name in target_names:
                if name == target_name:
                    print(name)
                    log.write(f"{time_signature}: {name} detected \n")
                    return True
            return False

    logfile_name = f"logfile_app_processed_scene_{query.vid_name}"
    for person in query.persons:
        logfile_name += f"_{person}"
    logfile_path = logfile_name + ".txt"
    log = open(f"{logfile_path}", "w") 

    processed_frames = []
    i = 0
    timestamp_last_recognition = -5000
    timestamp_last_detection = -2000
    last_detection_no = 0
    
    # for each scene
    while(capture.isOpened()):

        retval, frame = capture.read()
        if retval:
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
            processed_frame = frame.copy()
            # if 1s has passed since the last detection
            if (timestamp - timestamp_last_detection > 1000 or timestamp - timestamp_last_recognition > 2500):
                    timestamp_last_detection = timestamp
                    # detect faces
                    results_frame = model(frame)
                    print(len(results_frame[0].boxes))
                    # if the number of detected faces has changed since last detection or if 2.5s have passed since last recognition
                    if (len(results_frame[0].boxes) != last_detection_no or timestamp - timestamp_last_recognition > 2500):
                        last_detection_no = len(results_frame[0].boxes)
                        timestamp_last_recognition = timestamp
                        # for each box = detected face:
                        j = 0
                        for box in results_frame[0].boxes:
                            # save cropped face as image
                            image_path = Path(f'./temp/frame{i}_face{j}.jpg')
                            save_one_box(box.xyxy, frame, file=Path(f'./temp/frame{i}_face{j}.jpg'), BGR=True)
                            j += 1
                            if recognise_face(timestamp, image_path, query.persons) == True:
                                processed_frame = mark_box(box, processed_frame)
        else:
            break
        
        processed_frames.append(processed_frame)
        
        # testing condition, just for i first frames   
        if i > 2000:
            break

        i += 1
            
    fps = capture.get(cv2.CAP_PROP_FPS)
    procesessed_video_name = f"processed_app_scene_{query.vid_name}"
    for person in query.persons:
        procesessed_video_name += f"_{person}"
    procesessed_video_path = procesessed_video_name + ".mp4"
    processed_video = cv2.VideoWriter(procesessed_video_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (1280,720))

    for f in processed_frames:
        processed_video.write(f)

    processed_video.release()
    # TODO: check if the bucket exists and create it if it doesn't
    client.fput_object("processed-videos-bucket", procesessed_video_path, procesessed_video_path)

    log.close()
    # TODO: check if the bucket exists and create it if it doesn't
    client.fput_object("logfiles-bucket", logfile_path, logfile_path)

    answer = Answer(procesessed_video_path, logfile_path)

    return(answer)

def setup_minio():
    client = Minio("10.1.221.58:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
        http_client=urllib3.ProxyManager(
            "http://10.1.221.58:9000",
            timeout=urllib3.Timeout.DEFAULT_TIMEOUT,
            retries=urllib3.Retry(
                total=5,
                backoff_factor=0.2,
                status_forcelist=[500, 502, 503, 504],
            ),
        ),
    )
    return client

minio_client = setup_minio()
recognition_app = Flask(__name__)

@recognition_app.route('/recognise', methods=['GET'])
def get_recognised(): 
    data = request.get_json()
    query = Query(data["vid_name"], data["persons"])
    recognised = vid_recognise(query, minio_client)
    return recognised.toJSON()

@recognition_app.route("/health")
def healthcheck():
    return "Recognition app is up!"


            







