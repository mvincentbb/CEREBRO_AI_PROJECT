import pandas as pd
import seaborn as sns
import numpy as np
from IPython import get_ipython
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
import cv2
import os
import pathlib
import shutil
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
import random
from sklearn.preprocessing import LabelEncoder
import pickle




root = '/mnt/sdb1/vincent/ST25/script/side_project/'
# Base directory containing the folders of images
base_input_directory = root + 'data_256'
base_output_directory =  root + 'data_256_processed'

# Ensure the base output directory exists
if not os.path.exists(base_output_directory):
    os.makedirs(base_output_directory)

def preprocess_images(base_input_directory, base_output_directory):
    images = []
    labels = []
    label_names = os.listdir(base_input_directory)

    # Loop over each label
    for label in label_names:
        folder_path = os.path.join(base_input_directory, label)
        if os.path.isdir(folder_path):
            output_folder = os.path.join(base_output_directory, label)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Loop over each image file
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(folder_path, filename)
                    # Read the image
                    img = cv2.imread(file_path)
                    if img is None:
                        continue

                    # Resize the image
                    resized = cv2.resize(img, (256, 256))

                    # Grayscale the image
                    grayscaled = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                    # Flatten the image and append to the list
                    images.append(grayscaled.flatten())
                    labels.append(label)

                    # Save the processed image to the corresponding output subfolder
                    cv2.imwrite(os.path.join(output_folder, filename), grayscaled)

    return np.array(images), np.array(labels), label_names

# Path to the dataset
images, labels, label_names = preprocess_images(base_input_directory, base_output_directory)

# Normalize the images
X = preprocessing.normalize(images, axis=1)

# Apply PCA for dimensionality reduction
n_components = 100  # Adjust this number based on the variance you want to retain
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)




# Load VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)



def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.flatten()

# Apply feature extraction to the dataset
def load_and_extract_features(base_directory):
    features = []
    labels = []
    label_names = os.listdir(base_directory)

    for label in label_names:
        folder_path = os.path.join(base_directory, label)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith('.jpg'):
                    file_path = os.path.join(folder_path, filename)
                    features.append(extract_features(file_path))
                    labels.append(label)
    return np.array(features), np.array(labels), label_names

dataset_path =  root + 'data_256_processed'
X, y, label_names = load_and_extract_features(dataset_path)


# The Features has been extracted, but to save space we decided to delete the output of the previous code as it was taking a space in the notebook
# ##### Knowing that feature extracting will add dimensions to our dataset which is sadly beyond our capacity in computation, we will do another dimensionality reduction


# Apply PCA for another dimensionality reduction
n_components = 100
pca = PCA(n_components=n_components)
X = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(len(X_train))

def augment_labeled_set(X_labeled, y_labeled, X_unlabeled, clf, threshold):
    probs = clf.predict_proba(X_unlabeled)
    max_probs = np.max(probs, axis=1)
    confident_indices = np.where(max_probs >= threshold)[0]
    if len(confident_indices) == 0:
        return X_labeled, y_labeled, X_unlabeled, False

    confident_samples = X_unlabeled[confident_indices]
    confident_labels = np.argmax(probs, axis=1)[confident_indices]
    X_labeled = np.vstack((X_labeled, confident_samples))
    y_labeled = np.concatenate((y_labeled, confident_labels))
    X_unlabeled = np.delete(X_unlabeled, confident_indices, axis=0)
    return X_labeled, y_labeled, X_unlabeled, True


# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
threshold_scores = {}  # Dictionary to store scores for each threshold

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    X_train_labeled, X_unlabeled, y_train_labeled, _ = train_test_split(
        X_train, y_train, test_size=0.8, random_state=42)
    clf = DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=50)
    clf.fit(X_train_labeled, y_train_labeled)

    threshold = 0.95
    for i in range(5):  # Five iterations of threshold adjustment
        X_train_labeled, y_train_labeled, X_unlabeled, augmentation_occurred = augment_labeled_set(
            X_train_labeled, y_train_labeled, X_unlabeled, clf, threshold)
        
        # Fit model if augmentation occurred
        if augmentation_occurred:
            clf.fit(X_train_labeled, y_train_labeled)
        
        # Record the accuracy for this threshold
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        if threshold not in threshold_scores:
            threshold_scores[threshold] = []
        threshold_scores[threshold].append(accuracy)
        
        # Dynamic threshold decrement
        threshold -= 0.05 if augmentation_occurred else 0.1
        threshold = max(threshold, 0.1)  # Prevent it from going too low

# Compute mean scores for each threshold
mean_scores = {t: np.mean(scores) for t, scores in threshold_scores.items()}
for threshold, mean_score in mean_scores.items():
    print(f"Mean accuracy for threshold {threshold:.2f}: {mean_score:.4f}")




with open('decision_tree_semi_supervised_model.pkl', 'wb') as file:
    pickle.dump(clf, file)


