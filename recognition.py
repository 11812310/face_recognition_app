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
import os

model = YOLO("model.pt")

class Query:
    def __init__(self, vid_name, persons):
        self.vid_name = vid_name
        self.persons = []
        parsed_persons = json.loads(json.dumps(persons)) 
        print("PERSONS TO RECOGNISE: ")
        for person in parsed_persons:
            parsed_person = json.loads(json.dumps(person))
            print(parsed_person['person'])
            self.persons.append(parsed_person['person'])

class Recognition:
    def __init__(self, frame_no, timestamp, target_persons_name):
        self.frame_no = frame_no 
        self.timestamp = timestamp
        self.target_persons_name = target_persons_name
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4) 

class Recognitions:
    def __init__(self, recognitions):
        self.recognitions = recognitions
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4) 

def vid_recognise(query: Query, client) -> Recognitions:
    
    # download the video on which the recognition will run from minio and save it locally
    vid_path = f"{query.vid_name}.mp4"
    client.fget_object("original-videos-bucket", vid_path, vid_path)

    capture = cv2.VideoCapture(vid_path)

    if (capture.isOpened() == False):
        return("Video hasn't opened properly")


    def recognise_face(frame_number, time_signature, image_path, target_names): 
        dfs = DeepFace.find(img_path=image_path, db_path="database", enforce_detection=False) # possibly choose different model
        if dfs and len(dfs[0]['identity']) > 0:
            name = dfs[0]['identity'].to_string(index=False).split('/')[1]
            for target_name in target_names:
                if name == target_name:
                    print(name)
                    return Recognition(frame_number, time_signature, name)
            return None

    recognitions = []
    i = 0
    timestamp_last_recognition = -5000
    timestamp_last_detection = -2000
    last_detection_no = 0
    
    # for each scene
    while(capture.isOpened()):

        retval, frame = capture.read()
        if frame is None:
            break
        else:
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
            # if 4s have passed since the last detection
            if (timestamp - timestamp_last_detection > 4000 or timestamp - timestamp_last_recognition > 10000):
                    timestamp_last_detection = timestamp
                    # detect faces
                    results_frame = model(frame)
                    print(len(results_frame[0].boxes))
                    # if the number of detected faces has changed since last detection or if 10s have passed since last recognition
                    if (len(results_frame[0].boxes) != last_detection_no or timestamp - timestamp_last_recognition > 10000):
                        last_detection_no = len(results_frame[0].boxes)
                        timestamp_last_recognition = timestamp
                        # for each box = detected face:
                        j = 0
                        for box in results_frame[0].boxes:
                            # save cropped face as image
                            image_path = Path(f'./temp/frame{i}_face{j}.jpg') #TODO: check the possibility of always saving in the same file ?
                            save_one_box(box.xyxy, frame, file=Path(f'./temp/frame{i}_face{j}.jpg'), BGR=True) 
                            recognised = recognise_face(i, timestamp, image_path, query.persons)
                            if recognised != None:
                                recognitions.append(recognised)
                            j += 1
        i += 1

    recognitions = Recognitions()

    return(recognitions)

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
recognition = Flask(__name__)

@recognition.route('/recognise', methods=['GET'])
def get_recognised(): 
    data = request.get_json()
    query = Query(data["vid_name"], data["persons"])
    recognitions = vid_recognise(query, minio_client)
    return recognitions.toJSON()

@recognition.route("/health")
def healthcheck():
    return "Recognition function is available!"

if __name__ == "__main__":
   recognition.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

            







