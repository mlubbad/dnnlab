from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

#import pandas as pd
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix


#plt.ion()  

#use_gpu = torch.cuda.is_available()
#if use_gpu:
#    print("Using CUDA")


data_dir = '/home/dnnlab/muhammed/data'
model_path = '/home/dnnlab/muhammed/MnasNet/saved_model/MnasNet_best_model.pt'
TRAIN = 'test'
VAL = 'test'
TEST = 'test'
batch_size = 16
num_workers = 4
num_epochs = 250


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get the names of the ten classes
#class_names = testgen.class_indices.keys()

def plot_heatmap(y_true, y_pred, class_names, ax, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        square=True, 
        xticklabels=class_names, 
        yticklabels=class_names,
        fmt='d', 
        cmap=plt.cm.Blues,
        cbar=False,
        ax=ax
    )
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

#plot_heatmap(true_classes, scratch_pred_classes, class_names, ax1, title="Custom CNN")    
#plot_heatmap(true_classes, vgg_pred_classes, class_names, ax2, title="Transfer Learning (VGG16) No Fine-Tuning")    
#plot_heatmap(true_classes, vgg_pred_classes_ft, class_names, ax3, title="Transfer Learning (VGG16) with Fine-Tuning")    

#fig.suptitle("Confusion Matrix Model Comparison", fontsize=24)
#fig.tight_layout()
#fig.subplots_adjust(top=1.25)
#plt.show()

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally. 
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
}

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x), 
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL, TEST]
	}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    for x in [TRAIN, VAL, TEST]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

for x in [TRAIN, VAL, TEST]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))
    
print("Classes: ")
class_names = image_datasets[TEST].classes
print(image_datasets[TEST].classes)

since = time.time()
avg_loss = 0
avg_acc = 0
loss_test = 0
acc_test = 0
y_true = []
y_preds = []
    

test_batches = len(dataloaders[TEST])
print("Evaluating model")
print('-' * 10)
model = torch.hub.load('pytorch/vision:v0.13.0', 'mnasnet0_75', pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.13.0', 'googlenet', pretrained=True)
#model = models.googlenet(pretrained=True, progress=True)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
#model = models.squeezenet1_0(pretrained=True)
# Freeze training for all layers
#for param in model.features.parameters():
#    param.require_grad = False
# Newly created modules have require_grad=True by default
"""
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
model.classifier = nn.Sequential(*features) # Replace the model classifier
"""
"""
num_classes = len(class_names)
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
model.num_classes = num_classes
"""
"""
num_features = model.classifier.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.classifier = nn.Linear(num_features, len(class_names))
#model = model.to(device)
"""
"""
model.classifier[-1] = nn.Linear(1280, len(class_names))
"""
"""
num_features = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_features, len(class_names))
"""
model.classifier[-1] = nn.Linear(1280, len(class_names))

model.load_state_dict(torch.load(model_path))

criterion = torch.nn.CrossEntropyLoss()



#image=Image.open(path_to_data)
#image=image.resize((224,224))
#I = np.asarray(image)
#imshow(I)
try:
    model.eval()
except AttributeError as error:
    print(error)

for i, data in enumerate(dataloaders[TEST]):
    if i % 100 == 0:
        print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

    inputs, labels = data
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    #preds = torch.argmax(outputs.data, 1)
    loss = criterion(outputs, labels)

    #pred = outputs.cpu().data.numpy().argmax()
    #print(np.array(preds).tolist())
    #print(np.array(labels.data).tolist())
        #print(preds.round().reshape(-1).detach())
        #print(labels.data)
        #print(preds.cpu())
    #print(labels.data.cpu())
        #print(cf_matrix)
    loss_test += loss.data
    acc_test += torch.sum(preds == labels.data)
    for x in preds:
        y_preds.append(x)
    for y in labels.data:
        y_true.append(y)
    #print('accuracy of the model on test set : ',(((outputs.data.round().reshape(-1) == labels.data).sum())/float(outputs.data.shape[0])).item(),"%")
    plt.show()
    del inputs, labels, outputs, preds
    torch.cuda.empty_cache()

#print("\rClassification Report:\n", classification_report(labels.data.cpu(), preds.cpu()))
#cf_matrix = confusion_matrix(preds.cpu(),labels.data.cpu())
#sns.heatmap(cf_matrix, cmap='coolwarm', annot=True, linewidth=1, fmt="d")
#print(y_preds)
#print(y_true)

avg_loss = loss_test / dataset_sizes[TEST]
avg_acc = acc_test / dataset_sizes[TEST]
    
elapsed_time = time.time() - since
print()
print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
print("Avg loss (test): {:.4f}".format(avg_loss))
print("Avg acc (test): {:.4f}".format(avg_acc))
print('-' * 10)

#fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(20, 10))
fig, (ax1) = plt.subplots(1, 1,figsize=(20, 10))
#ax3 = plt.figure(figsize=(20, 10))

#plot_heatmap(true_classes, scratch_pred_classes, class_names, ax1, title="Custom CNN")    
#plot_heatmap(true_classes, vgg_pred_classes, class_names, ax2, title="Transfer Learning (VGG16) No Fine-Tuning")    
#plot_heatmap(y_true, y_preds, class_names, ax3, title="Transfer Learning (VGG16) with Fine-Tuning")    
plot_heatmap(y_true, y_preds, class_names, ax1, title="Transfer Learning (MnasNet0_75)")    

fig.suptitle("Confusion Matrix Model Chart", fontsize=24)
#fig.tight_layout()
#fig.subplots_adjust(top=1.25)
plt.show()

#for x in image_datasets[TEST]:
    #load_and_test(model_path, x, labels)
#    imshow(x)
