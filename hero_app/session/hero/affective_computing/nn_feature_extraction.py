"""
Converted from: nn_feature_extraction.ipynb
Original path: C:\Users\ratul\PycharmProjects\H-E-R-O-System\hero-monitor\affective_computing\nn_feature_extraction.ipynb
"""


# Skipped magic command (Cell 1):

# !wget -O embedder.tflite -q https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite


# Cell 2
from sklearn.preprocessing import LabelEncoder
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import config
from imutils import paths
import pickle
import random
import os

import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from hero.affective_computing.get_pipe_data import get_pipe_data
import cv2
import keras
import numpy as np
from hero.affective_computing.point_cloud import FaceCloud

from tqdm.notebook import tqdm


# Cell 3
random.seed(101)
# load the ResNet50 network and initialize the label encoder
print("[INFO] loading network...")
model = keras.models.load_model('../models/AffectInceptionResNetV3.keras')

# model = keras.models.load_model("data/checkpoints/checkpoint.keras")
le = None


# Cell 4
image_shape = (224, 224, 3)

inputs = keras.Input(shape=image_shape, name="image_input")
x = model.get_layer("rescale") (inputs)
x = model.get_layer("resnet50v2")(x)[0]
x = model.get_layer("global_pool")(x)
x = model.get_layer("feature_vector")(x)
new_model = keras.Model(inputs, x, name="AffectNN")
new_model.trainable = False

new_model.summary()
output_size = np.prod(new_model.output.shape[1:])


# Cell 5
# loop over the data splits
for split in (config.VAL, config.TRAIN):
	# grab all image paths in the current split
	print("[INFO] processing '{} split'...".format(split))
	p = os.path.sep.join([config.BASE_PATH, split])
	
	imagePaths = list(paths.list_images(p))
	# randomly shuffle the image paths and then extract the class
	# labels from the file paths
	random.shuffle(imagePaths)
	labels = [p.split(os.path.sep)[-2] for p in imagePaths]
	# if the label encoder is None, create it
	le = LabelEncoder()
	le.fit(labels)
	# open the output CSV file for writing
	
	csvPath = os.path.sep.join([config.BASE_CSV_PATH, f"{split}_data.csv"])
	csv = open(csvPath, "w")
	csv.write(",".join(["path", "class"] + [f"col_{idx}" for idx in range(output_size+3)]))
	csv.write("\n")
	
    # loop over the images in batches
	image_idx = 0
	for (b, i) in enumerate(tqdm(range(0, len(imagePaths), config.BATCH_SIZE))):
		# extract the batch of images and labels, then initialize the
		# list of actual images that will be passed through the network
		# for feature extraction
		batchPaths = imagePaths[i:i + config.BATCH_SIZE]
		batchLabels = le.transform(labels[i:i + config.BATCH_SIZE])
		batchImages = []
		# loop over the images and labels in the current batch
		for imagePath in batchPaths:
			# load the input image using the Keras helper utility
			# while ensuring the image is resized to 224x224 pixels
			image = load_img(imagePath, target_size=(224, 224))
			image = img_to_array(image)
			# preprocess the image by (1) expanding the dimensions and
			# (2) subtracting the mean RGB pixel intensity from the
			# ImageNet dataset
			image = np.expand_dims(image, axis=0)
			image = preprocess_input(image)
			# add the image to the batch
			batchImages.append(image)
        
        # pass the images through the network and use the outputs as
		# our actual features, then reshape the features into a
		# flattened volume
		batchImages = np.vstack(batchImages)
		features_1 = new_model.predict(batchImages, batch_size=config.BATCH_SIZE, verbose=0)
		features_2 = model.predict(batchImages, batch_size=config.BATCH_SIZE, verbose=0)
		
		features = np.concatenate([features_1, features_2], axis=1)
		features = features.reshape((-1, output_size+3))
		
		features = np.asarray(features, np.float16)
		# print(features.dtype)

		# loop over the class labels and extracted features
		for idx, (label, vec) in enumerate(zip(batchLabels, features)):
			# construct a row that exists of the class label and
			# extracted features
			vec = ",".join([str(v) for v in vec])
			csv.write(f"{batchPaths[idx]},{label},{vec}\n")
			
	# close the CSV file
	csv.close()
# serialize the label encoder to disk
f = open(config.LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()


# Cell 6
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True, num_faces=1, )
detector = vision.FaceLandmarker.create_from_options(options)

base_options = python.BaseOptions(model_asset_path='embedder.tflite')
options = vision.ImageEmbedderOptions(base_options=base_options, l2_normalize=True, quantize=True)
embedder = vision.ImageEmbedder.create_from_options(options)

for data_set in (config.TRAIN, config.TRAIN):
	
	image_size = (224, 224)
	shape_data = np.empty((0, 49))
	blend_data = np.empty((0, 52))
	embedding_data = np.empty((0, 1024))
	labels = np.empty((0, 1))
	net_data = pd.read_csv(f"training_data/{data_set}_data.csv")
	
	feature_count = net_data.shape[1]-2
	net_data.columns = ["path", "class"] + [f"col_{idx}" for idx in range(feature_count)]
	# print(net_data.head())
	net_data = net_data.set_index("path")
	
	fail_paths = []
	for im_path in tqdm(net_data.index):
		img_array = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
		# get image as mediapipe image
		img_mp = mp.Image(data=img_array, image_format=mp.ImageFormat.SRGB)
		embedding_result = embedder.embed(img_mp)
		face_landmarks, blend_feature, _ = get_pipe_data(detector, img_mp)
		if face_landmarks is not None:
			face = FaceCloud(face_landmarks)
			face.preprocess()
			shape_feature = face.create_shape_feature()
			shape_data = np.append(shape_data, np.reshape(shape_feature, (1, -1)), axis=0)
			blend_data = np.append(blend_data, np.reshape(blend_feature, (1, -1)), axis=0)
			embedding_data = np.append(embedding_data, embedding_result.embeddings[0].embedding.reshape(1, -1), axis=0)
		else:
			fail_paths.append(im_path)
	
	face_data = pd.DataFrame(
		data=np.concatenate([shape_data, blend_data, embedding_data], axis=1), 
		columns=([f"shape_{idx}" for idx in range(shape_data.shape[1])]+
				[f"blend_{idx}" for idx in range(blend_data.shape[1])]+
				[f"embedding_{idx}" for idx in range(embedding_data.shape[1])]))
	
	net_data_2 = net_data[np.logical_not(net_data.index.isin(fail_paths))]
	face_data = face_data.set_index(net_data_2.index)
	
	fuse_data = net_data_2.join(face_data)
	fuse_data.index = range(fuse_data.shape[0])
	fuse_data.to_csv(f"training_data/fuse_data_{data_set}.csv", index=False)
	print(fuse_data.shape)
