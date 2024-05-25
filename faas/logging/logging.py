from pathlib import Path
from flask import Flask, request
import json
from minio import Minio
import urllib3
import os

#TODO: pottentialy also parse detections json and log them to another file
#TODO: also, potentially log statistics based on both those jsons to a third file
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
    def __init__(self, logfile_name):
        self.logfile_name = logfile_name
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4) 

def log(query: Query, client) -> Answer:
    
    logfile_name = f"logfile_app_processed_scene_{query.vid_name}"
    for person in query.persons:
        logfile_name += f"_{person}"
    logfile_path = logfile_name + ".txt"
    log = open(f"{logfile_path}", "w") 

    for recognition in Query.recogitions:
        log.write(f"{recognition.timestamp}: {recognition.target_persons_name} detected \n")

    log.close()
    # TODO: check if the bucket exists and create it if it doesn't
    client.fput_object("logfiles-bucket", logfile_path, logfile_path)

    answer = Answer(logfile_path)

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
logging = Flask(__name__)

@logging.route('/log', methods=['GET'])
def get_logged(): 
    data = request.get_json()
    query = Query(data["vid_name"], data["recognitions"]) #TODO: adjust the recognition JSON
    marked = log(query, minio_client)
    return marked.toJSON()

@logging.route("/health")
def healthcheck():
    return "Logging function is available!"

if __name__ == "__main__":
   logging.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

            







