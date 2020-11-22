# FishDetector

## Overview

This project is to detect the fishes with the deep learning model. To get the best accuracy, in this project
Faster_RCNN_RESNET_50, 101, 152 models are trained with the custom object(fish) training data and estimated their own 
speed and accuracy.

## Structure

- src

    The source code to detect the fishes
    
- utils

    * The deep learning model
    * The source code for management of folders and files in this project

- app

    The main execution file
    
- requirements

    All the dependencies for this project
    
- settings

    The several settings including the path
    
## Installation

- Environment

    Ubuntu 18.04, Python 3.6

- Installing dependencies.
    ```
        pip3 install -r requirements.txt
    ``` 

  

## Execution

- Please navigate to this project directory and run the following command in the terminal.

    ```
        python3 app.py
    ```
