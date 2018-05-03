# over-the-air_speech_recogniztion_attack
--------------------------------------------------------------------------------

## Introduction
The course final project for UCLA EE209AS Winter 2018 - Special Topics in Circuits 
and Embedded Systems: Security and Privacy for Embedded Systems, Cyber-Physical 
Systems, and the Internet of Things by Professor. Mani Srivastava. In this project, 
we use deep learning (audio U-Net) to build model remove electronic noise and air 
noise during adversarial example transmission over the air. The contribution of 
this project are:

* make the adversarial example attack can transmit over-the-air, which eventually be a practical attack.
* found audio U-Net is also a possible defense for adversarial example attack due to strong ability to remove noise.

For more problem details, please go to
[my personal website](https://weikunhan.github.io).

## Requirements and Dependencies
Recommended use Anaconda to create the individual environment for this project 
and use following code to install dependencies:
```
conda install -c conda-forge tensorflow 
conda install -c conda-forge tqdm
conda install -c conda-forge librosa
conda install -c conda-forge sox
```
The following packages are required (the version numbers that have been tested 
are given for reference):

* Python 2.7 or 3.6
* Tensorflow 1.0.1
* Numpy 1.12.1
* Librosa 0.5.0
* tqdm 4.11.2 (only for preprocessing datasets)
* SoX 14.4.2 (only for preprocessing datasets)
* pysox 1.2.7 (brew install sox)
