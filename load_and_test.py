import torch
import torchvision.models as models

from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
from os import listdir
from numpy import asarray
import numpy as np

import torch.nn as nn

import shutil
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_one_image(I, model):
    '''
    I - 28x28 uint8 numpy array
    '''

    # test phase
    try:
        model.eval()
    except AttributeError as error:
    	print(error)

    ### 'dict' object has no attribute 'eval'
    #model.load_state_dict(checkpoint['state_dict'])
    ### now you can evaluate it
    model.eval()

    # convert image to torch tensor and add batch dim
    #batch = torch.tensor(I).unsqueeze(0)
    tensor = torch.from_numpy(I)
    batch = tensor.permute(2, 0, 1).unsqueeze(0)

    # We don't need gradients for test, so wrap in 
    # no_grad to save memory
    #with torch.no_grad():
    #    batch = batch.to(device)

    # forward propagation
    output = model( batch.type(torch.FloatTensor) )

    # get prediction
    #output = torch.argmax(output, 1)

    return output

# functions to show an image


def imshow(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))

def main(path_to_data, model_path):
    #VGG16_v2-OCT_Retina_complete_dataset.pt
    #file_path='/home/server/muhammed/saved_model/vgg_model/VGG16_v2-best.pt'
    #file_path='/home/server/muhammed/saved_model/vgg_model/VGG16_v2-OCT_Retina_complete_dataset.pt'
    file_path=model_path
    #model = torch.hub.load('pytorch/vision:v0.12.0', 'vgg16_bn', pretrained=True, verbose=False)
    model = models.vgg16_bn(pretrained=True)
    # Freeze training for all layers
    for param in model.features.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, 6)]) # Add our layer with 4 outputs
    model.classifier = nn.Sequential(*features) # Replace the model classifier
    #print(model)
    #model.eval()
    model.load_state_dict(torch.load(file_path))


    #model = torch.load('/home/server/muhammed/saved_model/vgg_model/VGG16_v2-best.pt')
    image=Image.open(path_to_data)
    image=image.resize((224,224))
    I = np.asarray(image)
    #I = np.ndarray(image)
    #I = np.transpose(I, (64, 3, 3, 3))
    #imshow(I)
    output = test_one_image(I, model)
    labels = ['bilimplant', 'dentium', 'dyna', 'implance', 'megagen', 'straumann']
    print(labels)
    print(output)
    output = torch.argmax(output, 1)
    print(labels[output])



def parse_args():
  parser = argparse.ArgumentParser(description="Single Image Classifier")
  parser.add_argument("--test_data_path", required=True, help="Path to data")
  parser.add_argument("--model_path", required=True, help="Path to model")
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  main(args.test_data_path, args.model_path)
