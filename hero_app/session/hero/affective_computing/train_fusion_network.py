"""
Converted from: train_fusion_network.ipynb
Original path: C:\Users\ratul\PycharmProjects\H-E-R-O-System\hero-monitor\affective_computing\train_fusion_network.ipynb
"""


# Cell 1
import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Cell 2
train_feature_df = pd.read_csv("training_data/fuse_data_train_set.csv")
val_feature_df = pd.read_csv("training_data/fuse_data_val_set.csv")


# Cell 3
train_features = train_feature_df.copy()

train_labels = train_features.pop('class')

val_features = val_feature_df.copy()
val_labels = val_features.pop('class')

print(train_features.shape[1])
print(val_features.shape[1])


# Cell 4
subsets = ["blend", "shape", "col", "embedding"]
col_names = pd.Series([])

for subset in subsets:
    col_names = pd.concat([col_names, pd.Series(train_features.columns[pd.Series(train_features.columns).str.startswith(subset)])])

train_features_subset = train_features[col_names]
val_features_subset = val_features[col_names]
print(train_features_subset.shape)


# Cell 5
x_train = train_features_subset.values
x_train = StandardScaler().fit_transform(x_train) # normalizing the features
x_val = val_features_subset.values
x_val = StandardScaler().fit_transform(x_val) # normalizing the features

train_features_norm = pd.DataFrame(x_train,columns=train_features_subset.columns)
val_features_norm = pd.DataFrame(x_val,columns=val_features_subset.columns)


# Cell 6
component_count = min(100, len(train_features_subset.columns))
pca_features = PCA(n_components=component_count)
principal_components_train = pca_features.fit_transform(x_train)
principal_components_val = pca_features.fit_transform(x_val)


# Cell 7
print('Explained variation per principal component: {}'.format(pca_features.explained_variance_ratio_))
np.sum(pca_features.explained_variance_ratio_)

principal_components_train = pd.DataFrame(data = principal_components_train, columns=[f"pca_{idx}" for idx in range(component_count)])
principal_components_val = pd.DataFrame(data = principal_components_val, columns=[f"pca_{idx}" for idx in range(component_count)])


# Cell 8
print(principal_components_train.shape, principal_components_val.shape)


# Cell 9
normalizer = keras.layers.Normalization(axis=-1)
normalizer.adapt(principal_components_train.values)
encoding = keras.layers.CategoryEncoding(num_tokens=3, output_mode="one_hot")

def construct_model(learning_rate=0.005):
    model = keras.Sequential([
        keras.layers.GaussianNoise(0.01),
        layers.Dense(128, activation="relu"),
        layers.Dense(200, activation="relu"),
        layers.Dense(128, activation="relu"), 
        layers.Dropout(0.2, name="dropout_regularisation"), # Regularize with dropout
        layers.Dense(3, activation="relu"),
    ])

  # preprocessed_inputs = preprocessing_head(inputs)
  # result = body

    model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
      loss=keras.losses.BinaryCrossentropy(from_logits=True),
      metrics=[keras.metrics.CategoricalAccuracy()],
    )
    return model


# Cell 10
base_epochs = 10
train_labels_1ht = encoding(train_labels)
val_labels_1ht = encoding(val_labels)


# Cell 11
fusion_model = construct_model(learning_rate=1e-5)
train_history = fusion_model.fit(x=principal_components_train, y=train_labels_1ht, epochs=20, validation_data=(principal_components_val, val_labels_1ht))


# Cell 12
# fusion_model.save("data/FusionNetwork2.h5")
fusion_model.save_weights("data/FusionWeights.weights.h5", overwrite=True)
keras.saving.save_model(fusion_model, 'data/FusionNetwork2.keras')

acc = train_history.history['categorical_accuracy']
val_acc = train_history.history['val_categorical_accuracy']

loss = train_history.history['loss']
val_loss = train_history.history['val_loss']


# Cell 13
import matplotlib.pyplot as plt
epochs_range = range(1, 11)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Cell 14
# fusion_model_fine = keras.models.load_model('data/FusionNetwork.keras')
fusion_model_fine = construct_model()
fusion_model_fine.load_weights("data/FusionWeights.weights.h5")


# Cell 15
total_epochs = base_epochs

for rate in [5e-6, 2e-6, 1e-6]:
    total_epochs += 10
    fusion_model_fine.compile(
            optimizer=keras.optimizers.Adam(learning_rate=rate),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.CategoricalAccuracy()],
        )
    train_history = fusion_model_fine.fit(x=train_features_dict, y=train_labels, validation_data=(val_features_dict, val_labels),
                             epochs=total_epochs,
                             initial_epoch=train_history.epoch[-1])
    
    acc += train_history.history['categorical_accuracy']
    val_acc += train_history.history['val_categorical_accuracy']
    
    loss += train_history.history['loss']
    val_loss += train_history.history['val_loss']


# Cell 16
for rate in [5e-7, 2e-7]:
    total_epochs += 10
    fusion_model_fine.compile(
            optimizer=keras.optimizers.Adam(learning_rate=rate),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.CategoricalAccuracy()],
        )
    train_history = fusion_model_fine.fit(x=train_features_dict, y=train_labels, validation_data=(val_features_dict, val_labels),
                             epochs=total_epochs,
                             initial_epoch=train_history.epoch[-1])
    
    acc += train_history.history['categorical_accuracy']
    val_acc += train_history.history['val_categorical_accuracy']
    
    loss += train_history.history['loss']
    val_loss += train_history.history['val_loss']
