# Cerebro project

## Presentation of the project

## Your source code package in Scikit-learn and PyTorch

We have these files

- DT_With_Supervised_Learning.ipynb  : this is the Jupyter notebook for all the semi supervised decision model and all the test we have make
- Decision_tree_semi_supervid_.ipynb : notebook for the semi supervised decision tree model

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

For the Decision tree
After change the "root" variable which is the folder where the dataset for training will be. 
        
```bash
python decision_tree_supervised.py
```
        
This command will generate a file name decision_tree_suppervised_model.pkl which is our model

    

## Instructions on how to run the pre-trained model on the provided sample test dataset

To run the pretrain model  on the sample dataset named `data_256_samples` :

For decision tree

```bash
python test_dt_supervided_samples.py
```
Again you have to specify the root folder which is here the sample data will be and the pretrain model
For CNN

## Description on how to obtain the Dataset from an available download link

The original dataset can be obtain here : 

The dataset that contain only the 5000 images that we use can be obtain here :
