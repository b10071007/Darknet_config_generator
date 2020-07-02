# Darknet Neural Network Configuration Generator
Forked from Abdullahi Adamu


## Why darknet-config-generator ?
If you have used darknet for one of your projects, you also understand the pain of editing the config file when you want to modify your network, optimization, and image augmentation parameters only to realize you forgot to edit another parameter after commencing training (bummer). You will also understand the pain of editing the configuration file to run inference. I implemented this to allow me to describe my neural network in a keras-like fashion and have a darknet config file generated.

Give it a try with your next darknet project :-) 

#### How to install
* You can pip install network configuration generator as shown below:
```
pip install darknet-config-generator==1.0.1 
```

#### Where do I go from here?
- Introduction.ipynb -  Provides an Example Usage of the darknet config generator with YoloV3 Network.
- example_generated.cfg - The resulting configuration file generation from Introduction.ipynb
