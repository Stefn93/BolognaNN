import os
from shutil import copyfile
import pickle
import numpy as np
from decimal import Decimal

dataset='bologna_dataset_sparse/'
trainset = 'bologna_train_sparse/'
testset = 'bologna_test_sparse/'



category = ''
n = 0
k = 0
'''
labels = []
sorted_files = sorted(os.listdir(dataset))
for file in sorted_files:
        file = file.replace(".jpg", "")
        file = file.split('_')
        x = float(file[0].replace('X', ""))
        y = float(file[1].replace('Y', ""))
        labels.append((x, y))

#with open('labels.lbl', 'wb') as f:
        #pickle.dump(labels, f)


#train_length = (length/100) * 80
#test_length = (length/100) * 20
'''

with open('labels.lbl', 'rb') as f:
    labels = pickle.load(f)

index = -1
train_labels = []
test_labels = []
sorted_files = sorted(os.listdir(dataset))

for files in sorted_files:
    if(index == 39):
        index = 0
    else:
        index += 1

    if(index<=7 or (index>=12 and index <= 27) or (index >= 32 and index <=39)):
        copyfile(dataset + files, trainset + files)
        train_labels.append(labels[index])
    else:
        copyfile(dataset + files, testset + files)
        test_labels.append(labels[index])

with open('train_labels.lbl', 'wb') as f:
    pickle.dump(train_labels, f)

with open('test_labels.lbl', 'wb') as f:
    pickle.dump(test_labels, f)
#print('Folder ' + str(n) + ' completed! \n')







