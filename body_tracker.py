# using openCV and mediapipe
# to determine joints and head

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult, PoseLandmark


class BodyTracker:
    """
    my intended public interface: list of PoseLandmark objects, and the track_body() method
    the rest are private attributes and methods that handle the pipeline so the body-tracking works
    """

    def __init__(self):
        self.__IMPORTANT_LANDMARKS = (PoseLandmark.NOSE,
                                      PoseLandmark.LEFT_SHOULDER,
                                      PoseLandmark.RIGHT_SHOULDER,
                                      PoseLandmark.LEFT_ELBOW,
                                      PoseLandmark.RIGHT_ELBOW,
                                      PoseLandmark.LEFT_WRIST,
                                      PoseLandmark.RIGHT_WRIST)

        self.__detector = self.create_detector()
        self.__cap = cv2.VideoCapture(0)

        self.timestamp = 0
        self.landmarks = None

    def create_detector(self):
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            num_poses=1,

            # add this to specify that input is from live camera feed
            running_mode=VisionTaskRunningMode.LIVE_STREAM,
            result_callback=self.update_landmarks
        )

        return vision.PoseLandmarker.create_from_options(options)

    def get_important_landmarks(self, detection_result):
        if not detection_result.pose_landmarks:
            return None

        important_landmarks = []
        # there will always be only one subject
        for pose_landmarks in detection_result.pose_landmarks:
            for index in self.__IMPORTANT_LANDMARKS:
                important_landmarks.append(pose_landmarks[index])

        return important_landmarks

    def update_landmarks(self, result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.landmarks = self.get_important_landmarks(result)

    def track_body(self):
        ret, img = self.__cap.read()

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

        self.__detector.detect_async(image=mp_image, timestamp_ms=self.timestamp)
        self.timestamp+=1
