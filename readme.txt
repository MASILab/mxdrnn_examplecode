--------------- "Multi-path x-D Recurrent Neural Networks for Collaborative Image Classification" example code -------------

This is the code of experiments on mnist, our code is based on PyTorch. 

Note that:

1. the objective of this code show the details of our implementing experiments which cannot be fully written in the paper. 

2. This is only the example code, we make it concise. The more abundant well-orginized code with more datasets would be publicly available in the future. 

3. Even though we provide the test accuracy on every epoch, but what we reported in our paper is the test accuracy of the epoch with best accuracy on validation set. 


------------- usage of the code  -------------------

1. The configs are presented in mnist.yaml. If we want to train different model (e.g. basline model and MxDRNN + baseline), we can change the "model_name" and "save_path" in mnist.yaml file.

2. The main.py is the main python script. Use the command line 'python main.py' to run. 

3. RNN-related components are in 'crnn' folder. 

4. mnist_loader.py is the data loader file when using PyTorh. 
