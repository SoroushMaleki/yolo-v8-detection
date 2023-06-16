from roboflow import Roboflow
from ultralytics import YOLO
import cv2
import numpy as np


def get_train_dataset():
    rf = Roboflow(api_key="MGYgXmcMtRu7afOhjHpn")
    project = rf.workspace("k562").project("k1-feno8")
    dataset = project.version(6).download("yolov8")


class TargetDetector(object):
    def __init__(self, yolo_weight_path):
        """
        Initializes an instance of the TargetDetector class.

        Args:
            yolo_weight_path (str): The path to the YOLO weight file.
        """
        self.weight_path = yolo_weight_path
        self.detector_model = self.load_yolo_model()

    def load_yolo_model(self):
        """
        Loads the YOLO model using the specified weight file.

        Returns:
            model: The loaded YOLO model.
        """
        model = YOLO(self.weight_path)
        return model

    def detect(self, img_path):
        """
        Performs target detection on the image at the specified path using the YOLO model.

        Args:
            img_path (str): The path to the input image.

        Returns:
            list: A list of detection results. each entry is a dictionary with coordinates, size, class and confidence
            of the corresponding detection.
        """
        detection_info = []
        img = cv2.imread(img_path)
        results = self.detector_model.predict(source=img, save=False, save_txt=False)
        for detection in results[0]:
            x, y, w, h = detection.boxes.xywh[0].cpu().numpy().squeeze()
            class_id = detection.boxes.cls.cpu().numpy().squeeze()
            confidence = detection.boxes.conf.cpu().numpy().squeeze()
            detection_info.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'class': class_id,
                'confidence': confidence
            })
        return detection_info, img.shape

    @staticmethod
    def calc_relative_area(detection_info_list, class_id, image_shape):
        total_area = 0
        for detection in detection_info_list:
            if detection['class'] == class_id:
                area = detection['w'] * detection['h']
                total_area += area
        return total_area / (image_shape[0] * image_shape[1])

    @staticmethod
    def find_min_distance(detection_info):
        """
        Finds the minimum distance between the centers of two detections.

        Args:
            detection_info (list): A list of dictionaries containing detection information.

        Returns:
            float: The minimum distance between the centers of any two detections.
        """
        min_distance = 1000000  # Initialize with a large value

        for i in range(len(detection_info) - 1):
            center1_x = detection_info[i]['x'] + detection_info[i]['w'] / 2
            center1_y = detection_info[i]['y'] + detection_info[i]['h'] / 2

            for j in range(i + 1, len(detection_info)):
                center2_x = detection_info[j]['x'] + detection_info[j]['w'] / 2
                center2_y = detection_info[j]['y'] + detection_info[j]['h'] / 2

                distance = np.sqrt((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2)
                min_distance = min(min_distance, distance)

        return min_distance


yolo_path = "yolo_v8_weights.pt"
image_path = "/home/soroush/Documents/cybera/cv_detection/test_img.jpg"
detector = TargetDetector(yolo_weight_path=yolo_path)
detections, image_shape = detector.detect(image_path)

min_dist = detector.find_min_distance(detections)
print(min_dist)
