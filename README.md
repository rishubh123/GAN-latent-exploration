# GAN-latent-exploration
This repository contains all the codebase and experimentations for GAN latent exploration project. This code base is build on top of the original code base for latent-composition work (https://github.com/chail/latent-composition). 

To run the experiments follow these steps to setup the network weights and dependencies: 

# Installation
```
 - pip install matplotlib
 - pip install ipython 
 # basic py libs: if not installed 
 
 - pip install ninja
 - pip install lpips
 - mkdir -p resources/dlib
 
 # face landmarks model
 - wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
 - mv shape_predictor_68_face_landmarks.dat.bz2 resources/dlib
 - bunzip2 resources/dlib/shape_predictor_68_face_landmarks.dat.bz2
 
 # identity loss model from pixel2style2pixel
 # Note: gdown is not working, have to find another way to download the pretrain model 
 - gdown --id 1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn # pretrained model 
 - mkdir -p resources/psp
 - mv model_ir_se50.pth resources/psp 
```

# Experiments 
```
 Run the code by running python3 infer_main.py
```
