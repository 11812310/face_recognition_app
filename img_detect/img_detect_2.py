from ultralytics import YOLO
# import torch
import cv2

model = YOLO("model.pt")

image_path = "image.jpg"

results = model(image_path)

image = cv2.imread(image_path)

def mark_boxes(results_boxes, fr):
    for box in results_boxes:
        top_left_corner = (int(box.xyxy.tolist()[0][0]), int(box.xyxy.tolist()[0][1]))
        bottom_right_corner = (int(box.xyxy.tolist()[0][2]), int(box.xyxy.tolist()[0][3]))
        blue = (255, 0, 0)
        thickness = 2

        cv2.rectangle(fr, top_left_corner, bottom_right_corner, blue, thickness)
    return fr

marked_image = mark_boxes(results[0].boxes, image)
cv2.imwrite("detect_test_5.jpg", marked_image)

