# CEREBRO_AI_PROJECT
## Introduction
This project aims to explore the application of both classical machine learning and deep learning techniques for image classification. Specifically, it focuses on how Decision Trees and Convolutional Neural Networks (CNNs) can be used to classify images of different venues from the Places365-Standard dataset. The challenges include handling high-dimensional image data, ensuring model robustness against overfitting, and optimizing computational efficiency in the training and inference phases.

## Methodologies
### Dataset:
The dataset used is Places365-Standard, containing around 1.8 million images organized into 365 scene categories. For our purposes, we selected five classes: Hotel room, Library indoor, Art Gallery, Banquet Hall, and Supermarket. Images were preprocessed by resizing to 256x256, converting to grayscale, normalizing pixel values, and applying data augmentation.

## Decision Trees
### Supervised Learning:
Initially, hyperparameter tuning was performed, followed by feature extraction using the VGG16 model and principal component analysis (PCA). This approach improved the model's accuracy to 0.769.

### Semi-Supervised Learning:
A similar approach was taken with semi-supervised learning, where a small amount of labeled data and a larger amount of unlabeled data were used. The accuracy achieved was 0.705.

### Convolutional Neural Networks (CNNs):
The CNN model consisted of five layers, each with a convolutional layer, activation layer (ReLU), pooling layer (max-pooling), batch normalization, and dropout (rate of 0.25). Data augmentation techniques such as random cropping, flipping, rotating, and color-jittering were applied during preprocessing. The CNN achieved an accuracy of 0.77.

## Results
Decision Trees: The supervised learning model achieved an accuracy of 0.769, and the semi-supervised learning model achieved an accuracy of 0.705.
CNN Model: The CNN model, with its robust architecture and preprocessing techniques, achieved an accuracy of 0.75.

## Conclusion
The report explores the performance and applicability of classical and deep learning techniques for image classification. Detailed methodologies, preprocessing steps, model architectures, and hyperparameter tuning are discussed, providing insights into the strengths and limitations of Decision Trees and CNNs in handling complex image datasets. This comprehensive investigation highlights the importance of feature extraction, dimensionality reduction, and appropriate model selection in achieving high accuracy in image classification tasks.

## Links:
Group Members:
1- Vincent De Paul: https://github.com/mvincentbb
2- Julien Zabiolle: https://github.com/Juzab
3- Oussama Hedjar: https://github.com/OussamaHedjar

Project: https://github.com/mvincentbb/CEREBRO_AI_PROJECT/tree/main
Sample data: https://github.com/mvincentbb/CEREBRO_AI_PROJECT/blob/main/data_256_sample.7z
