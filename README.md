# domain-adaptation-cpath

## About

Implementation of the paper "Bridging Domain Gaps in Computational Pathology: A Comparative Study of Adaptation Strategies"

## Abstract

Due to the high variability of Hematoxylin and Eosin (H&E)-stained Whole Slide Images (WSIs), hidden stratification and batch effects, generalizing beyond the training distribution is one of the main challenges in Deep Learning (DL) for Computational Pathology (CPath). But although DL depends on large volumes of diverse and annotated data, it is common to have a significant number of annotated samples from one or multiple source distributions, and another partially annotated or unlabelled dataset representing a target distribution for which we want to generalize, the so-called Domain Adaptation (DA). In this work, we focus on the task of generalizing from a single source distribution to a target domain. We evaluate three different DA strategies, namely FixMatch, CycleGAN, and a self-supervised feature extractor and show that DA is still a challenge in CPath.

## Clone this repository

To clone this repository open a terminal window and type:

```$ git clone git@github.com:joao-nunes/domain-adaptation-cpath.git```

##Install environment and dependencies

```
$ conda env create --file=environment.yaml
```
