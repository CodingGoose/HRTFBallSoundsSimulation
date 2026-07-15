import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult, PoseLandmark


IMPORTANT_LANDMARKS = [PoseLandmark.NOSE,
                       PoseLandmark.LEFT_SHOULDER,
                       PoseLandmark.RIGHT_SHOULDER,
                       PoseLandmark.LEFT_ELBOW,
                       PoseLandmark.RIGHT_ELBOW,
                       PoseLandmark.LEFT_WRIST,
                       PoseLandmark.RIGHT_WRIST]


def get_important_landmarks(detection_result):
    important_landmarks = []
    # there will always be only one subject
    for pose_landmarks in detection_result.pose_landmarks:
        for index in IMPORTANT_LANDMARKS:
            important_landmarks.append(pose_landmarks[index])

    return important_landmarks


def print_results(result:PoseLandmarkerResult, output_image:mp.Image, timestamp_ms:int):
    print(f"PoseLandmarkerResult: {get_important_landmarks(result)}")

# Creating PoseLandmarker object
base_options = python.BaseOptions(model_asset_path='../pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    num_poses=1,
    # add this to specify that input is from live camera feed
    running_mode=VisionTaskRunningMode.LIVE_STREAM,
    result_callback=print_results)

# detector = vision.PoseLandmarker.create_from_options(options)

i=0
with vision.PoseLandmarker.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0) # 0-stdin (my camera)
    while True:
        ret, img =  cap.read()
        i+=1

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        detector.detect_async(mp_image, i)
        print(i)


#seems to work nicely.