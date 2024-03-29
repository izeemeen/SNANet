# SNANet

This repository contains the code for the models described in the study:

"Artificial Neural Network Model with Astrocyte-Driven Short-Term Memory"

An article is available at https://www.mdpi.com/2313-7673/8/5/422


The SNANet model is based on the model from a study https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009246. 

## Dependencies
This code was developed on Ubuntu 20.04 with Pytorch 1.0+. To create a local virtual environment, use the following commands:

```bash
# create a new conda environment
conda create --name snanet python=3.8

# activate the conda environment
conda activate snanet

# install pytorch and torchvision
conda install pytorch torchvision

# clone the repo
git clone https://github.com/izeemeen/SNANet.git
```

## Pre-trained CIFAR-10 Models
The code for pre-training convolutional neural networks on a grayscale version of the CIFAR-10 image recognition dataset and the resulting feature embeddings used in our paper can be found in the ```SNANet_ROOT/cifar``` directory.

## Training
To train a model, run ```python main.py```. Any trained models will be saved to ```SNANet_ROOT/PARAM/{SNANet,RNN}```. Please run ```python main.py --h``` to find out more about the possible command-line options. For example, to train a SNANet model:

```python
 python main.py --model SNANet --noise-std 0.5 --l2-penalty 0.001 --pos-weight 5 --dprime 1.5 --patience 5 --seed 1
 ```

**Note:** Use the ```--no-cuda``` flag here and below if you want to run the cpu-only version of Pytorch.

## Evaluation
To test a model, run ```python experiment.py```. You must pass an appropriate model path for loading a trained model. Model results will be saved as pickle files to ```SNANet_ROOT/RESULT/{SNANet,RNN}```. Please run ```python experiment.py --h``` to see all possible command-line options. For example, to evaluate the SNANet model trained above:

```python
python experiment.py --model SNANet --model-path ./PARAM/SNANet/model_train_seed_1.pt --noise-std 0.5 --omit-frac 0.05 --seed 1
```

## Visual Behavior Analyses
We provide Jupyter notebooks that generate the figures in the paper in ```SNANet_ROOT/figures```.

