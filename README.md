# Roads Extraction from Satellite Images using Deep Learning
-------------------------------------------------------------
> ## Dataset:
Download the dataset from https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset \
and place the images data in the parent directory and inside a folder named 'data'

To get started, create an anaconda environment with python version 3.9.13
```python
conda create --name name_of_your_environment python=3.9.13
```
then activate the environment
```python
conda activate name_of_your_environment
```

Install the required libraries to run the project
```python
pip install -r requirements.txt
```
&nbsp;
> ## To run the project, do any one of the following options
1) run main.py with the proper environment
```python
/path/to/$CONDAENV/python.exe main.py
```
2) run Untitled1.ipynb in jupyter-notebook\

```python
pip install jupyter
jupyter-notebook # in command line
```
open the Untitled1.ipynb file and select Cell: Run All \
&nbsp;

> ## Code Description:
-----------------

Number of total images in the dataset: 6226
Number of total corresponging masks:   6226

The decided ratio to divide the dataset into training, testing and ratio is:
0.7 : 0.15 : 0.15

To divide the images dataset into 3 sections of training, testing and validation,
use of the `split-folders` library is suggested.

Install the split-folders in your conda environment:
```python
pip install split-folders
```
then import the splitfolders library:
```python
import splitfolders
```

> #### Splitfolders folder structure:
pythonProject/divided_data/\
├───test\
│   ├───images\
│   └───masks\
├───train\
│   ├───images\
│   └───masks\
└───val\
    ├───images\
    └───masks

#### Create proper folder structure for ease of use i.e.
by copying images to proper concerned directories\
pythonProject/divided_data/\
├───test_images/\
│   └───test/\
├───test_masks/\
│   └───test/\
├───train_images/\
│   └───train/\
├───train_masks/\
│   └───train/\
├───val_images/\
│   └───val/\
└───val_masks/\
    └───val/

After getting the proper folder structure,
verified if the images and masks correlate

For training, defined batch size is 16
And number of classes is 2 (roads area and non-roads area)

For data augmentation, ImageDataGenerator is used:
flipping random images and their corresponding masks
horizontally or vertically

Images and masks color mode is kept grayscale
and are resized to (128, 128)

For preprocessing part,
images are scaled down to 1
and as masks have only two values i.e roads(0) and non-roads(255),
we label encode them to 0 and 1 respectively by initially reshaping the 2D masks to 1D array
and again reshaping back to original 2D dimensions

Then masks are converted to categorical form

Then the architecture of u-net is define
steps per epoch is defined as number of training images / batch size
and validation steps per epoch is defined as number of validation images / batch size

Optimizer used is 'adam', loss function: 'binary_crossentropy', and metrics used is 'accuracy'

Number of epochs: 25

Then the model is saved to an hdf5 file for further use by not re-training the model again\
then the graphs concerning training and validation accuracy are plot together\
and for testing various testing images are plot and their predicted masks are plot

Later the Mean IoU is calculated\

![image](/output.png)  
