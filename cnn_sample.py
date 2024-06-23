#!/usr/bin/env python
# coding: utf-8

# ## CNN With Supervised Learning on a sample dataset
# 
# NOTE: The code has been optimized to work on Visual Code Studio.
# 
# To run this code, have the unziped folder "data_256_sample" in the same folder as the code ("cnn_sample.py") and also have the model file "CNN256.pt" in the same folder. And run all the boxes one after the other.
# This will :
# - define the model architecture
# - perform the testing on the testing set and return the corresponding loss, accuracy, precision, recall, and f1-score as well as plot the confusion matrix,
# 
# If you want to train the model on a the complete data_set, please refer to the code "cnn.ipynb".

# ### increase maximum display limit


# ### Import libraries :

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import shutil
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ### Define the model and import it's weights

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Layer 1: Convolutional, ReLU, MaxPooling, BatchNormalization, Dropout
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25)  # Adding dropout with p=0.25
        )
        
        # Layer 2: Convolutional, ReLU, MaxPooling, BatchNormalization, Dropout
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25)  # Adding dropout with p=0.25
        )
        
        # Layer 3: Convolutional, ReLU, MaxPooling, BatchNormalization, Dropout
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25)  # Adding dropout with p=0.25
        )
        
        # Layer 4: Convolutional, ReLU, MaxPooling, BatchNormalization, Dropout
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Dropout(0.25)  # Adding dropout with p=0.25
        )
        
        # Layer 5: Convolutional, ReLU, MaxPooling, BatchNormalization, Dropout
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Dropout(0.25)  # Adding dropout with p=0.25
        )
        
        # Adaptive pooling to ensure output size is (1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 5)  # because 5 classes

        #Softmax
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Forward pass through the layers
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        # Adaptive average pooling to (1, 1)
        out = self.avgpool(out)
        
        # Flatten for fully connected layers
        out = torch.flatten(out, 1)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.fc2(out)

        # Softmax
        # out = self.softmax(out)
        
        return out

# Create an instance of the CNN model
model = CNN()

if torch.cuda.is_available():
    model = model.cuda() #Move the model to GPU if available



model = CNN() 
model.load_state_dict(torch.load("./CNN256.pt"))
_= model.eval() #put model into evaluation mode


# ### Test Model on sample dataset

# Load sample dataset without any initial transforms
data_sample_dir = "./data_256_sample/"
dataset_sample = ImageFolder(data_sample_dir)

# preprocess
sample_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset_sample.transform = sample_transforms

#Define the size of the batches
batch_size=32

# Create DataLoaders for the training and test sets
dataloader_sample = DataLoader(dataset_sample, batch_size=batch_size, shuffle=False, pin_memory=True) # do not shuffle

#respecify it because it was specified in a def
criterion = nn.CrossEntropyLoss()

running_loss_test=0.0

# store all the predictions of the testing dataset to do a confusion matrix
all_preds = []
all_labels = []

with torch.no_grad(): #don't calculate gradiants because don't need to update the model weights
    for images, labels in dataloader_sample :
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        #loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss_test += loss.item()

        #prediction -> prepare for the confusion matrix and metrics
        _, preds = torch.max(outputs, dim=1) # predicted class indices for each example in the batch
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Convert lists to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)

# Calculate precision, recall, and F-score
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Loss:      {running_loss_test/len(dataloader_sample):.4f}')
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# Generate the confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset_sample.classes)
_=disp.plot(cmap=plt.cm.Blues)
plt.show()
plt.close()