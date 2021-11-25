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
<!---
# Results 
## Approach 1 Results 
### Editing results for eyeglass attribute editing, alpha: strength of edit operation
source: input image, inversion: inverted image from GAN
<img width="769" alt="eyeglasses1" src="https://user-images.githubusercontent.com/16732827/142574919-4a3da48f-3a52-4afa-9235-b03fc8cdcb7f.png">
<img width="773" alt="eyeglasses2" src="https://user-images.githubusercontent.com/16732827/142574966-0e673f43-e553-417d-adfa-cb1804e507d9.png">

### Editing results for hat attribute editing, alpha: strength of edit operation 
<img width="772" alt="hat" src="https://user-images.githubusercontent.com/16732827/142574991-dc502623-d4f4-406a-8296-83f3ebdda612.png">

### Editing results for smile attribute editing, alpha: strength of edit operation 
<img width="779" alt="smile1" src="https://user-images.githubusercontent.com/16732827/142575006-634f940a-4fe1-4cdf-a720-554ec4e49ee4.png">
<img width="770" alt="smile2" src="https://user-images.githubusercontent.com/16732827/142575009-8a825b47-ac5e-4599-8273-7c111b6bc22f.png">

## Approach 1 Results with identity loss 
### Editing results for eyeglasses, smile and hat attribute, alpha: stength of edit operation
<img width="771" alt="results_w_id" src="https://user-images.githubusercontent.com/16732827/142575722-3beb69f9-1762-496d-be3a-a84c8ace4b33.png">


## Approach 2 Results without idenity loss
### Synthetic data creation by augmenting the soruce image with hat and eyeglasses attributes
<img width="771" alt="data_synthesis" src="https://user-images.githubusercontent.com/16732827/142576957-639d29f4-9a48-4f5a-babd-2e0d32ac295e.png">


### Editing results for eyeglasses attribute, alpha: stength of edit
<img width="761" alt="eyeglasses1" src="https://user-images.githubusercontent.com/16732827/142576982-c74ce039-d5dc-445b-b1d5-82b50a09e0a8.png">
<img width="773" alt="eyeglasses2" src="https://user-images.githubusercontent.com/16732827/142576986-e19f6530-eb75-4b7b-b43e-29c362a3b921.png">


### Editing results for hat attribute, alpha: stength of edit operation
<img width="780" alt="hat" src="https://user-images.githubusercontent.com/16732827/142577001-29559150-fa40-456c-bb0c-e916a552e27c.png">

---> 
