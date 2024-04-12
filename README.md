# ERAV2_Main
Main repo with the required utils and other functions to train a model and check the outcomes

## main.py
File containing functions that are used to
- Run Training
- Run Evaluation
- Find Max LR for OneCyleLR
- Setup the required Optimizer, Loss Criteria and LR Scheduler
Some prerequisites
- Need to have output and temp folder for it to work
- output images / plots will be written to output folder
- temp is used to store the temp model
  
This can be run in 2 different modes
- By importing the file and running different functions in appropriate order to get the results
  - Look at sample implementation at https://github.com/ChintanShahDS/Assignment11
- By running as a python mail file passing the required params inline
  - python ERAV2_Main/main.py --num_epochs=3 --lr=0.01 --batchsize=512 --optimizer='SGD' --criterion='CrossEntropyLoss' --lrscheduler='OneCycleLR'

#### Supported hyperparameters
- Optimizer
  - SGD
  - Adam
- Loss Criterion
  - CrossEntropyLoss
- Scheduler
  - OneCyleLR
  - LROnPlateau
  - CosineAnnealingLR

## utils.py
File containing functions that are used for different kinds of tasks
- Visualization of data or plots like images from dataloader, gradcam outcomes, plots of accuracy curves
- Getting the Incorrect Predictions to see what should be done to improve the accuracy
- Train and Test data set data download, transform and setting up dataloaders (Currently only supported for Cifar10 dataset)
- Getting mean and Standard deviation (Not used)

## models folder
Contains the different model files
### init.py
- loads all the models inside the model folder

### resnet.py
- Has the basic Resnet18 and Resnet34 model structures
- Models that only use BasicBlock are implemented

### Watch out this space for future updates and more inclusions

