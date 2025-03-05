# Deep Learning GAN Fashion Design System

## Overview

The **Deep Learning GAN Fashion Design System** is a Generative Adversarial Network (GAN) based model designed to generate fashion-related images. It uses the **Fashion-MNIST** dataset to train a **generator** and **discriminator**, enabling the creation of realistic fashion items. This project leverages TensorFlow and Keras to build, train, and evaluate the GAN model.

## Features

* Uses Fashion-MNIST dataset for training.
* Implements a Generator model using convolutional layers with upsampling.
* Implements a Discriminator model with convolutional layers for classification.
* Trains using Binary Cross Entropy Loss and Adam Optimizer.
* Saves trained models for future image generation.
* Includes a callback function to save generated images during training.

Project Structure
```
├── data/                 # Fashion-MNIST dataset (loaded via TensorFlow Datasets)
├── models/               # Saved generator and discriminator models
├── images/               # Generated fashion images
├── fashion_gan.py        # Main script for training the GAN
├── requirements.txt      # Required Python packages
├── README.md             # Project documentation
```

## Installation

**Clone the repository:**
```bash
git clone https://github.com/Deepak250104/DeepLearning-GAN-Fashion.git
cd DeepLearning-GAN-Fashion
```
**Install dependencies:**
```bash
pip install -r requirements.txt
```
**Ensure you have TensorFlow installed:**
```bash
pip install tensorflow
```
## Usage

**1. Import Dependencies and Load Data**
* The dataset is loaded using TensorFlow Datasets and preprocessed (scaled to [0,1] range).

**2. Build GAN Components**
* Generator: Converts random noise into fashion images.
* Discriminator: Classifies images as real or fake.

**3. Train the GAN**
Run the training script:
```bash
python fashion_gan.py
```
Training progress will be saved, and generated images will be stored in the images/ directory.

**4. Generate New Fashion Items**
After training, load the generator model and generate images:
```python
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

generator = load_model('models/generator.h5')
random_input = np.random.normal(size=(16, 128, 1))
generated_images = generator.predict(random_input)

fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(10,10))
for i in range(4):
    for j in range(4):
        ax[i, j].imshow(generated_images[i * 4 + j].squeeze(), cmap='gray')
        ax[i, j].axis('off')
plt.show()
```

## Training Details

* Loss Function: Binary Cross Entropy

* Optimizers: Adam (learning rates: Generator - 0.0001, Discriminator - 0.00001)

* Epochs: Set to 100

* Batch Size: 128

## Model Saving & Loading

The trained models are saved as generator.h5 and discriminator.h5.

They can be reloaded for generating new images.

## Results & Performance

The model progressively improves in generating realistic fashion items.

The loss graph shows generator and discriminator loss decreasing over time.

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License.

## Author
Deepak


