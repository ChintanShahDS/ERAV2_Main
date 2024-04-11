'''Train CIFAR10 with PyTorch.'''
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchsummary import summary
from tqdm import tqdm

from models import *
from utils import *

# Fast AI method
def find_maxlr():
	from torch_lr_finder import LRFinder
	optimizer_lr = optim.Adam(mymodel.parameters(), lr=1e-7, weight_decay=1e-1)
	lr_finder = LRFinder(mymodel, optimizer_lr, criterion, device="cuda")
	lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
	__, maxlr = lr_finder.plot() # to inspect the loss-learning rate graph
	lr_finder.reset() # to reset the model and optimizer to their initial state
	print("max_LR:", maxlr)
	return maxlr

def train(model, device, trainloader, optimizer, criterion, scheduler):
    model.train()
    pbar = tqdm(trainloader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        # print("Size of data:", len(data), "Size of target:", len(target))
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        scheduler.step()

    train_acc= 100*correct/processed
    print('\nProcessed: {}, Len TrainLoader: {}'.format(processed, len(trainloader)))
    # train_losses.append(train_loss/len(trainloader))
    train_loss = train_loss/len(trainloader)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(trainloader.dataset),
        100. * correct / len(trainloader.dataset)))
    last_lr = scheduler.get_last_lr()
    print(f"Last computed learning rate: {last_lr}")
    print("LR Rate:", optimizer.param_groups[0]['lr'])

    return train_acc, train_loss

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            test_loss+=loss.item()
            total += target.size(0)

            correct += GetCorrectPredCount(output, target)

    print("target total:", total, "test_loader len:", len(test_loader.dataset))
    test_loss /= total
    test_acc = (100. * correct / total)
    # test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, total, 100. * correct / total))

    return test_acc, test_loss

def setupTrainingParams(initialModelPath, optimizer_name, criterion_name, scheduler_name, base_lr):

    if lrsch_name == 'OneCycleLR':
        mymodel = torch.load(initialModelPath)
        maxlr = find_maxlr()

    mymodel = torch.load(initialModelPath)

    criterion = nn.CrossEntropyLoss()
    if criterion_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        print("This Loss Criteria is currently not supported")
        raise Exception()
        
    optimizer = optim.SGD(mymodel.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=5e-4)
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(mymodel.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(mymodel.parameters(), lr=base_lr)
    else:
        print("This Optimizer is currently not supported")
        raise Exception()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    if lrsch_name == 'OneCycleLR':
        pct_start = 0.3
        base_momentum = 0.85
        max_momentum = 0.9
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=maxlr, div_factor=10,
                                                    final_div_factor=10, steps_per_epoch=len(train_loader),
                                                    epochs=num_epochs, pct_start=max_lr_epoch/num_epochs,
                                                    three_phase=False, anneal_strategy='linear')
    elif lrsch_name == 'LROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.1, threshold_mode='rel', verbose=True)
    elif lrsch_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        print("This LR Scheduler is currently not supported. Supported Schedulers are 'OneCycleLR', 'LROnPlateau' and 'CosineAnnealingLR'")
        raise Exception()
        
    return mymodel, optimizer, criterion, scheduler


def runTraining(train_loader, test_loader, initialModelPath, optimizer_name, criterion_name, scheduler_name, num_epochs=20, base_lr=0.01):

    mymodel, optimizer, criterion, scheduler = setupTrainingParams(initialModelPath, optimizer_name, criterion_name, scheduler_name, base_lr)
    # Data to plot accuracy and loss graphs
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(1, num_epochs+1):
        print('\nEpoch: %d' % epoch)
        train_acc, train_loss = train(mymodel, device, train_loader, optimizer, criterion, scheduler)
        test_acc, test_loss = test(mymodel, device, test_loader, criterion)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        
    return mymodel, train_losses, test_losses, train_accs, test_accs

def main():
    print("Start of the main module")
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer')
    parser.add_argument('--criterion', default='CrossEntropyLoss', type=str, help='loss criteria')
    parser.add_argument('--lrscheduler', default='OneCycleLR', type=str, help='lr scheduler')
    # parser.add_argument('--resume', '-r', action='store_true',
                        # help='resume from checkpoint')
    args = parser.parse_args()
    print("Arguments parsing complete")

    # getting all arguments
    num_epochs = args.num_epochs
    lr = args.lr
    optimizer_name = args.optimizer
    criterion_name = args.criterion
    lrsch_name = args.lrscheduler

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    # ds_mean, ds_std = get_mean_and_std(dataset)

    # Data
    print('==> Preparing data..')
    transform_train, transform_test = trainTestTransforms(ds_mean, ds_std)


    batchsize = 512

    kwargs = {'batch_size':batchsize, 'shuffle': True, 'num_workers': 4, 'pin_memory': True}

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, **kwargs)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, **kwargs)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    visualizeData(train_loader, 20, classes)

    # Model
    print('==> Building model..')
    mymodel = ResNet18()
    mymodel = mymodel.to(device)
    summary(mymodel, input_size=(3, 32, 32))
    torch.save(mymodel, 'InitialModel.pth')
    if device == 'cuda':
        mymodel = torch.nn.DataParallel(mymodel)
        cudnn.benchmark = True

    drawLossAccuracyPlots(train_losses, train_accs, test_losses, test_accs)

    numImages = 10
    images, nonMatchingLabels, incorrectPreds = incorrectOutcomes(mymodel, device, test_loader, numImages)
    showIncorrectPreds(numImages, images, incorrectPreds, nonMatchingLabels,classes, mymodel, gradCam=False)    

    showIncorrectPreds(numImages, images, incorrectPreds, nonMatchingLabels,classes, trained_model=mymodel, gradCam=True)
