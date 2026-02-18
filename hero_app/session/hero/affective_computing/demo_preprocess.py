"""
Converted from: demo_preprocess.ipynb
Original path: C:\Users\ratul\PycharmProjects\H-E-R-O-System\hero-monitor\affective_computing\demo_preprocess.ipynb
"""


# Cell 1
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from hero.affective_computing.get_pipe_data import get_pipe_data
import cv2
from hero.affective_computing.point_cloud import FaceCloud


# Cell 2
im_path = "sample_images/Happy_26.png"
# im_path = "Affective_Computing/Sample_Images/Ben Glasses.png"

# get image as RGB array
img_array = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
# get image as mediapipe image
img_mp = mp.Image(data=img_array, image_format=mp.ImageFormat.SRGB)

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True, num_faces=1, )
detector = vision.FaceLandmarker.create_from_options(options)
face_landmarks, blend_data, _ = get_pipe_data(detector, img_mp)


# Cell 3
face = FaceCloud(face_landmarks)
face.preprocess(scale=True, demo=True)
