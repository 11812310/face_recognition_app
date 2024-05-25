from pathlib import Path
import torch
import cv2
from flask import Flask, request
import json
from minio import Minio
import urllib3
import os

class Query:
    def __init__(self, vid_name, persons, recognitions):
        self.vid_name = vid_name
        self.persons = []
        parsed_persons = json.loads(json.dumps(persons)) 
        for person in parsed_persons:
            parsed_person = json.loads(json.dumps(person))
            print(parsed_person['person'])
            self.persons.append(parsed_person['person'])
        self.recognitions = []
        parsed_recognitions = json.loads(json.dumps(recognitions)) 
        for recognition in parsed_recognitions:
            pr = json.loads(json.dumps(recognition))
            recognition = Recognition(pr['frame_no'], pr['timestamp'], pr['box'], pr['target_persons_name'])
            self.recognitions.append(recognition)

class Recognition:
    def __init__(self, frame_no, timestamp, box, target_persons_name):
        self.frame_no = frame_no 
        self.timestamp = timestamp
        self.box = box
        self.target_persons_name = target_persons_name
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4) 

class Answer:
    def __init__(self, output_vid_name):
        self.output_vid_name = output_vid_name
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4) 

def vid_mark(query: Query, client) -> Answer:
    
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

    i = 0
    next_rec = 0
    fps = capture.get(cv2.CAP_PROP_FPS)
    procesessed_video_name = f"processed_app_scene_{query.vid_name}"
    for person in query.persons:
        procesessed_video_name += f"_{person}"
    procesessed_video_path = procesessed_video_name + ".mp4"
    processed_video = cv2.VideoWriter(procesessed_video_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (1280,720))
    
    # for each scene
    while(capture.isOpened()):
        retval, frame = capture.read()
        if retval:
            break
        else:
            while i == query.recognitions[next_rec].frame_no:
                mark_box(frame, query.recognitions[next_rec].box)
                next_recognition += 1
            processed_video.write(frame)
            i += 1

    processed_video.release()
    # TODO: check if the bucket exists and create it if it doesn't
    client.fput_object("processed-videos-bucket", procesessed_video_path, procesessed_video_path)

    answer = Answer(procesessed_video_path)

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
marking = Flask(__name__)

@marking.route('/mark', methods=['GET'])
def get_marked(): 
    data = request.get_json()
    query = Query(data["vid_name"], data["recognitions"]) #TODO: adjust the recognition JSON
    marked = vid_mark(query, minio_client)
    return marked.toJSON()

@marking.route("/health")
def healthcheck():
    return "Marking function is available!"

if __name__ == "__main__":
   marking.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

            







