import os
from shutil import copyfile
import pickle
import numpy as np
from decimal import Decimal

dataset='bologna_dataset_sparse/'
trainset = 'bologna_train_sparse/'
testset = 'bologna_test_sparse/'


def getint(name):
    basename = name.partition('.')
    parts = name.partition('_')
    return int(parts[0])

def generateLabels(folderPath, fileName):
    files = os.listdir(folderPath)
    files.sort(key=getint)

    labels = []
    for file in files:
        file = file.replace(".jpg", "")
        file = file.split('_')
        x = float(file[1].replace('X', ""))
        y = float(file[2].replace('Y', ""))
        labels.append((x, y))

    with open(fileName + '.lbl', 'wb') as f:
        pickle.dump(labels, f)


def checkMapping(folderPath, labelFile):
    files = os.listdir(folderPath)
    files.sort(key=getint)

    with open(labelFile, 'rb') as f:
        labels = pickle.load(f)

    i=0
    for file in files:
        print(file + ' --> ' + str(labels[i]))
        i+=1


def splitDataset(datasetFolder, trainsetFolder, testsetFolder):
    index = -1

    for files in os.listdir(datasetFolder):
        if(index == 39):
            index = 0
        else:
            index += 1

        if(index<=7 or (index>=12 and index <= 27) or (index >= 32 and index <=39)):
            copyfile(datasetFolder + files, trainsetFolder + files)
        else:
            copyfile(datasetFolder + files, testsetFolder + files)

def changeNames(folder):
    i = 0
    for file in os.listdir(folder):
        os.rename(folder+file, folder + str(i) + '_' + file)
        i+=1

#splitDataset(dataset, trainset, testset)
#changeNames(trainset)
#changeNames(testset)
#generateLabels(trainset, 'train_labels')
#generateLabels(testset, 'test_labels')
#checkMapping(trainset, 'train_labels.lbl')
#checkMapping(testset, 'test_labels.lbl')
