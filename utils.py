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
import torchvision.transforms.v2 as transforms
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

        plt.title("Pred:" + pred + "\nGT:" + gt, color='#ff0000')

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

    createImagePreds(num_images, images, labels[0:num_images], labels[0:num_images], classes, '/content/output/DataLoaderImage.jpg')

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
    plt.savefig('/content/output/AccuracyPlots.jpg', bbox_inches='tight')
    plt.show()

def showIncorrectPreds(numImages, images, incorrectPreds, nonMatchingLabels,classes):
    imagename = '/content/output/IncorrectPreds.jpg'
    createImagePreds(numImages, images, incorrectPreds, nonMatchingLabels, classes, imagename)

def showGradCam(numImages, images, incorrectPreds, nonMatchingLabels,classes, model, target_layers):
    ds_mean = (0.4914, 0.4822, 0.4465)
    ds_std = (0.247, 0.243, 0.261)
    _misclassified_batch = np.array(images)

    # target_layers = [model.layer3[-1]]
    input_tensor = torch.from_numpy(_misclassified_batch)

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(t) for t in nonMatchingLabels]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    mean_R, mean_G, mean_B = ds_mean
    std_R, std_G, std_B = ds_std

    fig = plt.figure(figsize=(24, 10))

    i = 0
    for actual in nonMatchingLabels[0:numImages]:
        # print(i, (2*i)+2)
        ax = plt.subplot(4, 10, (2 * i)+2)
        plt.axis('off')

        cam_result = grayscale_cam[i, :]

        _misclassified_batch[i, 0, :, :] = (_misclassified_batch[i, 0, :, :] * std_R) + mean_R
        _misclassified_batch[i, 1, :, :] = (_misclassified_batch[i, 1, :, :] * std_G) + mean_G
        _misclassified_batch[i, 2, :, :] = (_misclassified_batch[i, 2, :, :] * std_B) + mean_B

        visualization = show_cam_on_image(_misclassified_batch[i].transpose((1, 2, 0)), cam_result, use_rgb=True, image_weight=0.6)

        plt.imshow(visualization, cmap='jet')

        ax = plt.subplot(4, 10, (2 * i)+1)
        plt.axis('off')
        plt.imshow(_misclassified_batch[i].transpose((1, 2, 0)), cmap='jet')

        ax.set_title(f"actual: {classes[actual]} \n predicted: {classes[incorrectPreds[i]]}")
        i = i+1

    plt.savefig('/content/output/gradcam.jpg', bbox_inches='tight')
    plt.show()

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
        transforms.Pad(16, mean, 'constant'),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(scale=(0.125, 0.125), ratio=(1, 1), value=mean, inplace=False),
        transforms.CenterCrop(32),
        ])
        
    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean, std),
        ])
        
    return train_transforms, test_transforms

def getCifar10DataLoader(batchsize):
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
    # print("Classes:", classes)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
               # 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes
