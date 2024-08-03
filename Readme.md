# Character Recognizer

Character Recognizer is a Python-based project for recognizing characters from images using OpenCV and machine learning. This project involves training a model on a set of character images and then using that trained model to recognize characters in new images.

## Table of Contents
- [Character Recognizer](#character-recognizer)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Testing the Model](#testing-the-model)

## Introduction
The Character Recognizer project utilizes OpenCV for image processing and machine learning techniques to recognize characters in images. The project consists of two main parts:
1. Training a model using character images.
2. Using the trained model to recognize characters in new images.

## Features
- Training a character recognition model from a set of character images.
- Recognizing characters from new images using the trained model.
- Visualization of the training process and recognition results.

## Installation
To get started with the Character Recognizer, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/DevadattaP/Character_recognizer.git
    cd character-recognizer
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Training the Model
To train the model, run the `TrainModel.py` script with a training image containing characters.

```bash
python TrainModel.py
```
>[!NOTE]
> When the training program is run, segmented image will be shown, and you have to tell the machine what the highlighted letter/digit/character is by pressing corresponding key.

### Testing the Model
To recognize characters from a new image, run the `TestModel.py` script with the test image.

```bash
python TestModel.py
```

> [!TIP]
> You can change the image names in the both the python files, to test and train model on different images.