"""
Converted from: create_affective_dataset_V2.ipynb
Original path: C:\Users\ratul\PycharmProjects\H-E-R-O-System\hero-monitor\affective_computing\create_affective_dataset_V2.ipynb
"""


# Cell 1
import numpy as np
import pandas as pd
import os
import math
from scipy.io import savemat
import shutil
from PIL import Image


# Cell 2
np.random.seed(101)


# Cell 3
base_path_structured = "/Users/benhoskings/Documents/Datasets/FusionV2"
train_path_structured = os.path.join(base_path_structured, "train_set")
val_path_structured = os.path.join(base_path_structured, "val_set")


# Cell 4
emotions_affect_net = pd.Series(["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"])
# 3794, 1091 is given as contempt should be happy?
emotions_aff_wild = pd.Series(["Neutral","Anger","Disgust","Fear","Happy","Sad","Surprise","Other"])

processed_emotions = pd.Series(["Neutral", "Positive", "Negative"])

for set_name in ["train_set", "val_set"]:
    if not os.path.isdir(os.path.join(base_path_structured, set_name)):
        os.mkdir(os.path.join(base_path_structured, set_name))
        
    for em in pd.unique(pd.concat([processed_emotions])):
        if not os.path.isdir(os.path.join(base_path_structured, set_name, em)):
            os.mkdir(os.path.join(base_path_structured, set_name, em))


# Cell 5
def get_sample_ids(emotions, counts, max_size=None):
    # counts = [24882, 3750, 3803, 6378, 134414, 74874, 25459, 14090]
    label_count = dict(zip(emotions, counts))
    
    if max_size:
        max_size = min([max_size, min(label_count.values())])
    else:
        max_size = min(label_count.values())
        
    ids1 = np.empty((max_size, 0), np.int32)
    ids2 = np.empty((0, 1), np.int32)
    
    for idx, emotion in enumerate(emotions):
        file_count = label_count[emotion]
        emIds = np.random.permutation(np.arange(file_count))[:max_size]
        start_idx = sum(counts[:idx])
        ids1 = np.append(ids1, np.expand_dims(emIds, axis=1), axis=1)
        ids2 = np.append(ids2, start_idx + emIds)
        
    return ids1, ids2, class_count

def num_string(num):
    if num != 0:
        return f"0000{int(num)}"[int(math.log10(num)):]
    else:
        return "00000"

def is_corrupted(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify() # verify that it is, in fact an image
        return False
    except:
        return True
    
def map_affectnet_emotion(emotion):
    emotion: np.dtypes.StrDType
    if emotion.astype(np.uint8) < 2:
        label = emotion
    elif emotion.astype(np.uint8) == 3 or emotion.astype(np.uint8) == 7:
        label = 3
    else:
        label = 2
        
    return label

emotions_aff_wild = pd.Series(["Neutral","Anger","Disgust","Fear","Happy","Sad","Surprise","Other"])

def map_affectwild_emotion(emotion):
    emotion: np.dtypes.StrDType
    if emotion == 0:
        label = 0
    elif emotion == 4:
        label = 1
    elif emotion > 5:
        label = 3
    else:
        label = 2
        
    return label


# ======================================================================
# ## AffectNet Processing
# ======================================================================


# Cell 7
set_counts = {"train_set": 414797, "val_set": 5496}
# set_counts = {"val_set": 5496}
base_path_raw = "/Users/benhoskings/Documents/Datasets/AffectNet/Data"

for set_name, count in set_counts.items():
    base_path_set = os.path.join(base_path_raw, set_name)
    # logical array indicating if each instance has an image associated with it
    has_image = np.array([os.path.isfile(f"{base_path_set}/images/{idx}.jpg") for idx in range(count)])
    # logical array indicating if each instance has an emotion associated with it
    has_emotion = np.array([os.path.isfile(f"{base_path_set}/annotations/{idx}_exp.npy") for idx in range(count)])
    
    # create a dataframe to store values
    train_labels_affect_net = pd.DataFrame(index=range(count))
    train_labels_affect_net["has_image"] = has_image # update image array 
    train_labels_affect_net["has_emotion"] = has_emotion # update annotation array 
    
    # filter any instance which does not have an image AND a labelled emotion
    train_labels_affect_net = train_labels_affect_net.loc[
        np.logical_and(train_labels_affect_net["has_image"] == True, train_labels_affect_net["has_emotion"] == True)]
    
    # get image paths for all remaining instances 
    image_paths = [f"{base_path_set}/images/{idx}.jpg" for idx in train_labels_affect_net.index]
    affect_net_emotions = np.array(
        [map_affectnet_emotion(np.load(f"{base_path_set}/annotations/{idx}_exp.npy")) for idx in train_labels_affect_net.index],
        dtype=np.uint8)
    
    # assign image and emotion paths to dataframe 
    train_labels_affect_net["image_path"] = image_paths
    train_labels_affect_net["emotion"] = affect_net_emotions
    
    # set the index of the dataframe to the image path
    train_labels_affect_net = train_labels_affect_net.set_index("image_path")
    
    # Remove any duplicated rows 
    train_labels_affect_net = train_labels_affect_net[~train_labels_affect_net.index.duplicated(keep='first')]
    
    # Order by the emotions 
    train_labels_affect_net = train_labels_affect_net.sort_values(by=["emotion"])
    
    # Number of samples of each class
    affect_net_class_count = train_labels_affect_net.value_counts(subset=['emotion'])
    
    class_count = np.array(affect_net_class_count)
    class_labels = np.array([id for id in affect_net_class_count.index]).flatten()
    class_count = class_count[np.argsort(class_labels)]
    
    id1, id2, sample_count = get_sample_ids(emotions=processed_emotions, counts=class_count)
    train_subset = train_labels_affect_net.iloc[id2, :]
    
    class_counts = np.zeros((len(processed_emotions), 1))
    
    for im_path in train_subset.index:
        values = train_subset.loc[im_path]
        emotion_idx = int(values["emotion"])
        emotion = processed_emotions[emotion_idx]
        class_idx = class_counts[emotion_idx]
        sample_path = os.path.join(base_path_structured, set_name, emotion, "AN-" + num_string(class_idx.item()))
        # savemat(f"{sample_path}.mat", values.to_dict())
        shutil.copy(im_path, f"{sample_path}.png")
        class_counts[emotion_idx] += 1


# ======================================================================
# ## Aff-Wild-V2 Processing
# ======================================================================


# Cell 9
base_path_raw = "/Users/benhoskings/Documents/Datasets/Aff-Wild-V2/Provided"
label_path = os.path.join(base_path_raw, "Third ABAW Annotations/MTL_Challenge")
image_path_raw = os.path.join(base_path_raw, "Images")

for set_name in ["val_set"]:
    # read label values
    labels = pd.read_csv(os.path.join(label_path, set_name + ".txt"), index_col=0)
    labels = labels.loc[labels['expression'] >= 0]
    
    labels.expression = [map_affectwild_emotion(expression) for expression in labels.expression]
    labels = labels.sort_values(by=["expression"])
    
    corrupt = np.array([is_corrupted(os.path.join(image_path_raw, path)) for path in labels.index])
    labels = labels.loc[np.logical_not(corrupt)]
    labels = labels[~labels.index.duplicated(keep='first')]
    
    print(len(pd.unique(labels.index)))

    aff_wild_class_count = labels.value_counts(subset=['expression'])
    print(aff_wild_class_count)
    
    class_count = np.array(aff_wild_class_count, dtype=np.int64)
    class_labels = np.array([id[0] for id in aff_wild_class_count.index], dtype=np.uint16)
    class_count = class_count[np.argsort(class_labels)]
    id1, id2, sample_count = get_sample_ids(processed_emotions, class_count)
    train_subset = labels.iloc[id2, :]
    print(train_subset.value_counts(subset=['expression']))
    
    print(train_subset.head(10).to_string())
    class_counts = np.zeros((len(processed_emotions), 1))
    
    for im_path in train_subset.index:
        values = train_subset.loc[im_path]
        emotion_idx = int(values["expression"])
        emotion = processed_emotions[emotion_idx]
        class_idx = class_counts[emotion_idx]
        sample_path = os.path.join(base_path_structured, set_name, emotion, "AW-" + num_string(class_idx.item()))
        print(sample_path)
        # savemat(f"{sample_path}.mat", values.to_dict())
        shutil.copy(os.path.join(image_path_raw, im_path), f"{sample_path}.png")
        class_counts[emotion_idx] += 1
