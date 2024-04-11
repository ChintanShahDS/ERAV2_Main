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

import sys
sys.path.insert(0, '/content/ERAV2_Main')

from models import *
from utils import *

# Fast AI method
def find_maxlr(mymodel, train_loader):
    from torch_lr_finder import LRFinder
    criterion = nn.CrossEntropyLoss()
    optimizer_lr = optim.SGD(mymodel.parameters(), lr=1e-6, weight_decay=1e-1)
    # optimizer_lr = optim.Adam(mymodel.parameters(), lr=1e-7, weight_decay=1e-1)
    lr_finder = LRFinder(mymodel, optimizer_lr, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200)
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

def setupTrainingParams(initialModelPath, optimizer_name, criterion_name, scheduler_name, train_loader, num_epochs, base_lr):

    print(optimizer_name, criterion_name, scheduler_name)

    if (scheduler_name == 'OneCycleLR'):
        mymodel = torch.load(initialModelPath)
        maxlr = find_maxlr(mymodel, train_loader)

    mymodel = torch.load(initialModelPath)

    criterion = nn.CrossEntropyLoss()
    if (criterion_name=='ÇrossEntropyLoss'):
        criterion = nn.CrossEntropyLoss()
    else:
        print("This Loss Criteria is currently not supported")
        
    optimizer = optim.SGD(mymodel.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=5e-4)
    if (optimizer_name == 'SGD'):
        optimizer = optim.SGD(mymodel.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=5e-4)
    elif (optimizer_name == 'Adam'):
        optimizer = optim.Adam(mymodel.parameters(), lr=base_lr)
    else:
        print("This Optimizer is currently not supported")
        raise Exception()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    if (scheduler_name == 'OneCycleLR'):
        pct_start = 0.3
        base_momentum = 0.85
        max_momentum = 0.9
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=maxlr, div_factor=10,
                                                    final_div_factor=10, steps_per_epoch=len(train_loader),
                                                    epochs=num_epochs, pct_start=pct_start,
                                                    three_phase=False, anneal_strategy='linear')
    elif (scheduler_name == 'LROnPlateau'):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.1, threshold_mode='rel', verbose=True)
    elif (scheduler_name == 'CosineAnnealingLR'):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        print("This LR Scheduler is currently not supported. Supported Schedulers are 'OneCycleLR', 'LROnPlateau' and 'CosineAnnealingLR'")
        raise Exception()
        
    return mymodel, optimizer, criterion, scheduler


def runTraining(train_loader, test_loader, initialModelPath, optimizer_name, criterion_name, scheduler_name, device, num_epochs=20, base_lr=0.01):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mymodel, optimizer, criterion, scheduler = setupTrainingParams(initialModelPath, optimizer_name, criterion_name, scheduler_name, train_loader, num_epochs, base_lr)
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
    
# This is to be properly created
# Currently a temporary thing - Able to run as a Python main earlier
# But need to create such a way that can be run as functions or as main
def main(batchsize, num_epochs, base_lr, optimizer_name, criterion_name, scheduler_name):

    initialModelPath = '/content/temp/InitialModel.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    train_loader, test_loader, classes = getCifar10DataLoader(batchsize)

    visualizeData(train_loader, 20, classes)

    # Model
    print('==> Building model..')
    mymodel = ResNet18()
    mymodel = mymodel.to(device)
    torch.save(mymodel, initialModelPath)
    summary(mymodel, input_size=(3, 32, 32))

    print('==> Training model..')
    mymodel, train_losses, test_losses, train_accs, test_accs = runTraining(
        train_loader, test_loader, initialModelPath, optimizer_name, 
        criterion_name, scheduler_name, device, num_epochs=num_epochs, base_lr=base_lr)

    print('==> Accuracy plots..')
    drawLossAccuracyPlots(train_losses, train_accs, test_losses, test_accs)

    print('==> Incorrect outcomes..')
    numImages = 10
    images, nonMatchingLabels, incorrectPreds = incorrectOutcomes(mymodel, device, test_loader, numImages)
    showIncorrectPreds(numImages, images, incorrectPreds, nonMatchingLabels,classes)

    print('==> Incorrect outcomes explanation using GradCam..')
    showGradCam(numImages, images, incorrectPreds, nonMatchingLabels, classes, mymodel, [mymodel.layer4[-1]])
    
    print('==> End of the training and results..')

if __name__ == '__main__':
    print("Start of the main module")
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batchsize', default=20, type=int, help='Batch size')
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
    batchsize = args.batchsize
    base_lr = args.lr
    optimizer_name = args.optimizer
    criterion_name = args.criterion
    scheduler_name = args.lrscheduler
    
    main(batchsize, num_epochs, base_lr, optimizer_name, criterion_name, scheduler_name)
