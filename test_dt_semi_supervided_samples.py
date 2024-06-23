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
base_input_directory = root + 'data_256_sample'
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



# Apply PCA for another dimensionality reduction
n_components = 100
pca = PCA(n_components=n_components)
X = pca.fit_transform(X)


# # ### Split The Dataset

# # In[ ]:

# Load the model
with open('decision_tree_semi_supervised_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# # print(len(X_train))
y_test = y
X_test = X

# # ## Training
# # In this approach we choose to do hyperparameter tuning in a more effcient way makig our model able to find the best hyperparameters using grid search algorithm.

# # In[ ]:


# # Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)



# Now we predict the outcome

# In[ ]:


# Make predictions
# Test the model



# We use Label encode to transform our labels into a numeric vale, which is what is accepted in the classifier for decision tree in scikit learn.

# In[ ]:


y_test = le.fit_transform(y_test)


# ## Model Evaluation
# Now we evaluate our nodel using the metrics requested.
# Make predictions
y_pred = loaded_model.predict(X_test)
# In[ ]:
print(loaded_model.score(X_test, y_test))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Calculate F1-score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1-score: {f1}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
plt.close()

# Classification report
class_report = classification_report(y_test, y_pred, target_names=label_names)
print('Classification Report:')
print(class_report)
