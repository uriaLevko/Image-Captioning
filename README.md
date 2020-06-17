# Image-Captioning PyTorch 0.4
Project Overview
In this project, I created a neural network architecture to automatically generate captions from images, using the MS COCO dataset to train the network.

Project Instructions
The project is structured as a series of Jupyter notebooks that are designed in sequential order:
0_Dataset.ipynb
1_Preliminaries.ipynb
2_Training.ipynb
3_Inference.ipynb


LSTM Decoder - In the project, we pass all our inputs as a sequence to an LSTM. A sequence looks like this: first a feature vector that is extracted from an input image, then a start word, then the next word, the next word, and so on!

Embedding Dimension - The LSTM is defined such that, as it sequentially looks at inputs, it expects that each individual input in a sequence is of a consistent size and so we embed the feature vector and each word so that they are embed_size.
