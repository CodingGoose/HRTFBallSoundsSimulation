import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision

# head, shoulders, arms, & hands
IMPORTANT_LANDMARKS = [0, 11, 12, 13, 14, 15, 16]
def get_important_landmarks(detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    for pose_landmarks in pose_landmarks_list:
      important_landmarks=[]
      for index in IMPORTANT_LANDMARKS:
        important_landmarks.append(pose_landmarks[index])

    return important_landmarks

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:

      drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=pose_landmarks,
          connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
          landmark_drawing_spec=pose_landmark_style,
          connection_drawing_spec=pose_connection_style)

  return annotated_image



# STEP 1: Import the necessary modules.
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='../pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    num_poses=3)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("../images/daboys.jpg")

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

print(len(detection_result.pose_landmarks))

for landmark in detection_result.pose_landmarks:
    print(landmark)
# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

#openCV uses BGR instead of RGB for some reason
annotated_image=cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
annotated_image=cv2.resize(annotated_image,(640,480))

cv2.imshow('BAAAA', annotated_image)

cv2.waitKey(0)
# cv2.imwrite('images/outputs/output.jpg', annotated_image)
cv2.destroyAllWindows()

