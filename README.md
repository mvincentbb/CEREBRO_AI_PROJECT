# Cerebro project

## Presentation of the project

## Your source code package in Scikit-learn and PyTorch

We have these files

- [DT_With_Supervised_Learning.ipynb](https://github.com/mvincentbb/CEREBRO_AI_PROJECT/blob/main/DT_With_Supervised_Learning.ipynb)  : this is the Jupyter notebook for all the semi supervised decision model and all the test we have make
- [Decision_tree_semi_supervid_.ipynb](https://github.com/mvincentbb/CEREBRO_AI_PROJECT/blob/main/Decision_tree_semi_supervid_.ipynb) : notebook for the semi supervised decision tree model
- [data_256_sample.7z](https://github.com/mvincentbb/CEREBRO_AI_PROJECT/blob/main/data_256_sample.7z) is the sample data set to make the test on our pretrain model
- [cnn.ipynb](https://github.com/mvincentbb/CEREBRO_AI_PROJECT/blob/main/cnn.ipynb) is the notebook for our CNN model


## Requirements to run your Python code

- To run our code the following libraries is required
- pandas
- seaborn
- numpy
- matplotlib
- scikit-learn
- pillow
- opencv
- pytorch
- torchvision
- tensorflow

If you use conda  or [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) this one-liner will set up your environment with all necessary dependencies, assuming you have no issues with package conflicts or post-installation. 

```bash
conda create --name myenv python=3.8 && conda activate myenv && conda install pandas seaborn numpy matplotlib scikit-learn pillow opencv tqdm pytorch torchvision torchaudio cudatoolkit=10.2 tensorflow -c pytorch -c conda-forge && pip install pathlib
```

Here's a breakdown of the command:

- `conda create --name myenv python=3.8`: Creates a new Conda environment named `myenv` with Python 3.8.
- `conda activate myenv`: Activates the newly created environment.
- `conda install pandas seaborn numpy matplotlib scikit-learn pillow opencv tqdm pytorch torchvision torchaudio cudatoolkit=10.2 tensorflow -c pytorch -c conda-forge`: Installs the majority of the required packages directly via Conda, including TensorFlow and PyTorch with GPU support.
- `pip install pathlib`: Since `pathlib` is a standard library in Python 3.8 and does not need installation, this example uses `pip` to illustrate how you might install additional packages that are not available in Conda repositories.

Make sure to adjust the version of `cudatoolkit` to match your GPU drivers if necessary.

## Instruction on how to train/validate your model

### For the Decision tree supervised
After change the "root" variable which is the folder where the dataset for training will be. 
        
```bash
python decision_tree_supervised.py
```
        
This command will generate a file name decision_tree_suppervised_model.pkl which is our model
### For the Decision tree semi-supervised
After change the "root" variable which is the folder where the dataset for training will be. 
        
```bash
python decision_tree_semi_supervised.py
```
This command will generate a file name decision_tree_semi_supervised_model.pkl which is our  model
#### For the CNN With Supervised Learning

To run this code, have the unziped folder "data_256" in the same folder as the code ("cnn.ipynb"). And run all the boxes one after the other.
This will :
- preprocess the data,
- split it into training, validation and testing sets and print the number of images in each of those sets,
- Display an image from the dataset,
- display one batch from the training dataset,
- perfrom the training on the model and print, for each epoch, the loss on the training and validation sets and the accuracy on the validation set,
- plot the accuracy on the validation set on the number of epochs,
- plot the losses on the training and validation sets on teh number of epochs,
- perform the testing on the testing set and return the corresponding loss, accuracy, precision, recall, and f1-score as well as plot the confusion matrix,
- start a game where it will show you an image and you will have to compete against the model to find from which class this image is from.
- save the model to a file "CNN256.pt"

If you want to test the model on a sample data_set, please refer to the code "cnn_sample.ipynb".
NOTE: The code has been optimized to work on Visual Code Studio.

    

## Instructions on how to run the pre-trained model on the provided sample test dataset

To run the pretrain model  on the sample dataset named `data_256_sample` :

### For decision tree supervised

```bash
python test_dt_supervided_samples.py
```
Again you have to specify the root folder which is here the sample data will be and the pretrain model

### For decision tree  semi supervised 

```bash
python test_dt_semi_supervided_samples.py
```
Again you have to specify the root folder which is here the sample data will be and the pretrain model
### For CNN With Supervised Learning

To run this code, have the unziped folder "data_256_sample" and the model file ("CNN256.pt") in the same folder as the code ("cnn_sample.ipynb"). And run all the boxes one after the other.
This will :
- define the model architecture
- perform the testing on the testing set and return the corresponding loss, accuracy, precision, recall, and f1-score as well as plot the confusion matrix,

If you want to train the model on a the complete data_set, please refer to the code "cnn.ipynb".
NOTE: The code has been optimized to work on Visual Code Studio

## Description on how to obtain the Dataset from an available download link

The original dataset can be obtain [here](http://data.csail.mit.edu/places/places365/train256places365standard.tar)

The dataset that contain only the 5000 images that we use can be obtain [here](https://drive.google.com/file/d/1di2vovEidb91enydpqb1H5lWjDgSGwyA/view?usp=sharing)
