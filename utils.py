'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# import albumentations as A
# from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.init as init

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def GetIncorrectPreds(data, pPrediction, pLabels):
  images = []
  incorrectPreds = []
  nonMatchingLabels = []
  # print("pPrediction type:", type(pPrediction), "Shape:", pPrediction.shape)
  # print("pLabels type:", type(pLabels), "Shape:", pLabels.shape)
  preds = pPrediction.argmax(dim=1)
  indexes = pLabels.ne(pPrediction.argmax(dim=1))
  for image, pred, label in zip(data, preds, pLabels):
      if pred.ne(label):
          images.append(image.cpu())
          incorrectPreds.append(pred.cpu().item())
          nonMatchingLabels.append(label.cpu().item())

  # print("Incorrect Preds:", incorrectPreds, "Labels:", nonMatchingLabels)
  return images, incorrectPreds, nonMatchingLabels

def incorrectOutcomes(model, device, test_loader,reqData):
    model.eval()

    test_loss = 0
    correct = 0
    incorrectPreds = []
    nonMatchingLabels = []
    images = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            imageSet, incPred, nonMatchLabel = GetIncorrectPreds(data, output, target)
            nonMatchingLabels = nonMatchingLabels + nonMatchLabel
            incorrectPreds = incorrectPreds + incPred
            images = images + imageSet

            if len(incorrectPreds) > reqData:
              break

    return images, nonMatchingLabels, incorrectPreds

def imshowready(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def createImagePreds(numImages, images, preds, labels, classes, imagename):
    fig = plt.figure()
    for i in range(numImages):
        image = images[i]
        pred = classes[preds[i]]
        gt = classes[labels[i]]

        plt.subplot(2,int(numImages/2),i+1)
        # if gradCam:
            # image = showGradCam(image, trained_model)
        plt.imshow(imshowready(image))
        plt.axis('on')

        # ret = model.predict(data, batch_size=1)
        #print(ret)

        plt.title("Pred:" + pred + "\nGT:" + gt, color='#ff0000', fontdict={'fontsize': 12})

    plt.savefig(imagename, bbox_inches='tight')
    plt.show()
    return fig

def visualizeData(dataloader, num_images, classes):
    # get some random training images
    if num_images > len(dataloader):
        num_images = len(dataloader)
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images = images[0:num_images]

    createImagePreds(num_images, images, labels[0:num_images], labels[0:num_images], classes, 'DataLoaderImage.jpg')

def drawLossAccuracyPlots(train_losses, train_accs, test_losses, test_accs):
    fig = plt.figure()
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_accs)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_accs)
    axs[1, 1].set_title("Test Accuracy")
    plt.savefig('AccuracyPlots.jpg', bbox_inches='tight')
    plt.show()

def showGradCam(image, trained_model, target_layers, actual):
    target_layers = target_layers
    input_tensor = torch.Tensor(np.transpose(image, (2, 0, 1))).unsqueeze(dim=0)
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    # Hardcoded to True for now
    cam = GradCAM(model=trained_model, target_layers=target_layers)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    targets = [ClassifierOutputTarget(actual)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    return visualization

def showIncorrectPreds(numImages, images, incorrectPreds, nonMatchingLabels,classes, trained_model, gradCam=False):

    imagename = 'IncorrectPreds.jpg'
    if gradCam:
        imagename = 'IncorrectPredsWithGradCam.jpg'

    updatedImages = []
    if gradCam:
        for i in range(numImages):
            image = images[i]
            image = showGradCam(image, trained_model, [trained_model.layer3[-1]], nonMatchingLabels[i])
            updatedImages.append(image)
        images = updatedImages

    createImagePreds(numImages, images, incorrectPreds, nonMatchingLabels, classes, imagename)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


# Train data transformations
def getTrainTestTransforms(mean, std):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.5, 0.5), ratio=(1, 1), value=mean, inplace=False),
        transforms.Normalize(mean, std),
        ])
        
    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
        
    return train_transforms, test_transforms

def getCifar10DataLoader(batchsize)
    kwargs = {'batch_size':batchsize, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
    
    ds_mean = (0.4914, 0.4822, 0.4465)
    ds_std = (0.247, 0.243, 0.261)
    
    transform_train, transform_test = getTrainTestTransforms(ds_mean, ds_std)
    
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, **kwargs)

    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, **kwargs)

    classes = trainset.classes
    print("Classes:", classes)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
               # 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes
