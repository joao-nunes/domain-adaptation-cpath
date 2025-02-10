# domain-adaptation-cpath

## About

Implementation of the paper "Bridging Domain Gaps in Computational Pathology: A Comparative Study of Adaptation Strategies"

## Abstract

Due to the high variability of Hematoxylin and Eosin (H&E)-stained Whole Slide Images (WSIs), hidden stratification and batch effects, generalizing beyond the training distribution is one of the main challenges in Deep Learning (DL) for Computational Pathology (CPath). But although DL depends on large volumes of diverse and annotated data, it is common to have a significant number of annotated samples from one or multiple source distributions, and another partially annotated or unlabelled dataset representing a target distribution for which we want to generalize, the so-called Domain Adaptation (DA). In this work, we focus on the task of generalizing from a single source distribution to a target domain. We evaluate three different DA strategies, namely FixMatch, CycleGAN, and a self-supervised feature extractor and show that DA is still a challenge in CPath.

## Clone this repository

To clone this repository open a terminal window and type:

```$ git clone git@github.com:joao-nunes/domain-adaptation-cpath.git```

## Install environment and dependencies

```
$ conda env create --file=environment.yaml
```
## Usage

Train baseline feature extractor (resnet-34) supervised learning using only annotated data:

```
$ python3 run_baseline.py {command line arguments}
```

This script accepts the following command line arguments:

```
--train-library_file: library file with tile coords and target annotations for the training set
---val_library_file: library file with tile coords and target annotationss for the validation set
--train_json: jsonfile containing information for the training images
--valid_json: jsonfile containing infromation for the validation images
--resume: wether or not to resume training from last saved chekpoint
--ckpt: last saved checkppoint
"--num_claases: number fo classes"
"--in_features": number of features
"--batch_size": batch_size to consider. Note that for the training set. the actual batch size will be batch_size//8
"--workers": number of workers
"optimizer": optimizer for training the ML model
"save_dir": directory to save the results
""output": file name to save the model to
"mydataset": dataset class to load the data
"lr": learning rate
"epochs": num of epochs

```

Train the FixMatch feature extractor

```
$ python3 run-fixmatch.py
```

This script accepts the following command line arguments

```
--train_libraryfile: library file with tile coords and target annotations for the source training set (supervised loss)
--train_libraryfile2: library file with tile coords and target annotations for the target training set (unsupervised loss)
--val_libraryfile:  library file with tile coords and target annotationss for the source validation set (supervised loss)
--val_libraryfile2: library file with tile coords and target annotationss for the target validation set (unsupervised loss)
--train_json: jsonfile containing information for the source training images
--train_json2: jsonfile containing information for the target training images
--valid_json: jsonfile containing information for the source validation images
--valid_json2: jsonfile containing information for the target validation images
--resume: wether or not to resume training from last saved chekpoint
--ckpt: last saved checkpoint
--num_classes: number of classes
--in_features: number of features
--batch_size: batch size
--workers: number of workers
--optimizer: optimizer for training the ML model
--save_dir: directory to save the results
--output: file name to save the model to
--lr: learning rate
--epochs: number of training epochs
--thr: confidence threshold for consistency regularization
--lmbd: weight term of the unsupervised loss
--tau: temperature for sharpening of pseudo-labels
```
