# Complex Autoencoder

This repository contains an implementation of the concepts described in the paper "[*Complex-Valued Autoencoders for Object Discovery*](https://arxiv.org/abs/2204.02075)".

## Usage

### Installation
Run the executable `setup.sh` in order to install the conda environment, `pytorch`, `cudatoolkit` and `einops`. The script also installs the datasets (*2shapes*, *3shapes* and *MNIST_shapes*).

### Training
To train the network run the following command:
```
python ./main.py --task=train --dataset=2shapes --epochs=20
```
The above command launches the training on the dataset "*2shapes*" for 20 epochs.

The resulting network is saved in the directory `./models`.

### Evaluation
To evaluate the predictions of a network using the ARI score, run the following:
```
python .\main.py --task=evaluate --dataset=2shapes --model=./models/2shapes_99.pt
```
This command also plots the reconstruction and labeling prediction returned by the network.

In the folder `./models` you can find three trained models:
 - `2shapes_99.pt` trained using the *2shapes* dataset and having acquired an ARI score 0.99
 - `3shapes_99.pt` trained using the *3shapes* dataset and having acquired an ARI score 0.988
 - `MNIST_shapes_95.pt` trained using the *MNIST_shapes* dataset and having acquired an ARI score 0.95
