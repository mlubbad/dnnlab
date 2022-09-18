Transfer Learning for Computer Vision
==============================================
**Author**: `Mohammed Lubbad <https://mlubbad.github.io>`_
In this repository, you will learn how to train a convolutional neural network for
image classification using transfer learning. You can read more about the transfer
learning at `cs231n notes <https://cs231n.github.io/transfer-learning/>`__
Quoting these notes,
    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.
These two major transfer learning scenarios look as follows:
-  **Finetuning the convnet**: Instead of random initialization, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.

**Notes**
- We provide a code for trainning common ConvNets such like "vgg, resnet, 
  alexnet, densenet, efficientnet, googlenet, inception, mnasnet, mobilenet,
  squeezenet, resnext, swintransformer, visiontransformer, wideresnet, ..etc"

- You can find each ConvNet in a seperated directory, containning two python
  files "xxx_train.py", "xxx_test.py".

**Test**
Will calculate for you the following:
- Confusion Matrix
- Classification Report

**Get Started**
- Clone the repository 
- Define\create the data & saved_model folders
- Modify the path of data and saved model in the "xxx_train.py" file
- Run training code first "xxx_train.py"
- Modify the path of data and saved model in the "xxx_test.py" file
- Test the generated model by running "xxx_test.py" code

Goodluck :)
