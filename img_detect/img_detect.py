from ultralytics import YOLO
import torch
import cv2

model = YOLO("model.pt")

image_path = "image.jpg"

results = model(image_path)

image = cv2.imread(image_path)

# persons_boxes = []
# for each in results[0]:
#     print(each.names)
#     if each.names == "person":
#         persons_boxes.append(each.boxes)


for box in results[0].boxes:

    # if box.cls == 0:
        top_left_corner = (int(box.xyxy.tolist()[0][0]), int(box.xyxy.tolist()[0][1]))
        bottom_right_corner = (int(box.xyxy.tolist()[0][2]), int(box.xyxy.tolist()[0][3]))
        blue = (255, 0, 0)
        thickness = 2

        cv2.rectangle(image, top_left_corner, bottom_right_corner, blue, thickness)
        cv2.imwrite("detect_test_3.jpg", image)

