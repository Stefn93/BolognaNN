import numpy as np
import pickle

def normalizeValues(x, y, minX, minY, maxX, maxY):
    x = (x - minX) / (maxX - minX)
    y = (y - minY) / (maxY - minY)
    return x, y

def deNormalizeValues(x, y, minX, minY, maxX, maxY):
    x = x * (maxX - minX) + minX
    y = y * (maxY - minY) + minY
    return x, y

def serializeNorm(xy, str, minX, minY, maxX, maxY):
    labels = []
    for x, y in xy:
        labels.append(normalizeValues(x, y, minX, minY, maxX, maxY))

    with open(str + '.lbl', 'wb') as f:
        pickle.dump(labels, f)

def findMinMaxCoord(trainlabels, testlabels):
    minX = 999.0
    minY = 999.0
    maxX = 0.0
    maxY = 0.0

    for x, y in trainlabels:
        if x < minX:
            minX = x
        if x > maxX:
            maxX = x
        if y < minY:
            minY = y
        if y > maxY:
            maxY = y

    for x, y in testlabels:
        if x < minX:
            minX = x
        if x > maxX:
            maxX = x
        if y < minY:
            minY = y
        if y > maxY:
            maxY = y

    # print(str(minX) + ', ' + str(maxX) + ', ' + str(minY) + ', ' + str(maxY))
    return minX, minY, maxX, maxY


# Open the lbl files
with open('train_labels.lbl', 'rb') as fa:
    trainlabels = pickle.load(fa)
with open('test_labels.lbl', 'rb') as fe:
    testlabels = pickle.load(fe)

# Get coordinate bounds
minX, minY, maxX, maxY = findMinMaxCoord(trainlabels, testlabels)

# Serialization into files
serializeNorm(trainlabels, 'trainNorm', minX, minY, maxX, maxY)
serializeNorm(testlabels, 'testNorm', minX, minY, maxX, maxY)

# Check prints
with open('trainNorm.lbl', 'rb') as f:
    f = pickle.load(f)
    for i in range(len(f)):
        print(str(f[i]))
with open('testNorm.lbl', 'rb') as f:
    f = pickle.load(f)
    for i in range(len(f)):
        print(str(f[i]))
