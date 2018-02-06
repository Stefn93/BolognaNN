import os
from shutil import copyfile
import pickle
import numpy as np
from decimal import Decimal

dataset='bologna_dataset_sparse/'
trainset = 'bologna_train/'
testset = 'bologna_test/'



category = ''
n = 0
k = 0

labels = []
sorted_files = sorted(os.listdir(dataset))
for file in sorted_files:
        file = file.replace(".jpg", "")
        file = file.split('_')
        x = float(file[0].replace('X', ""))
        y = float(file[1].replace('Y', ""))
        labels.append((x, y))

with open('labels.lbl', 'wb') as f:
        pickle.dump(labels, f)

np.asarray(labels)

#train_length = (length/100) * 80
#test_length = (length/100) * 20

'''
index = -1

for files in os.listdir(dataset + dirs):
    if(index == 39):
        index = 0
    else:
        index += 1

    if(index<=7 or (index>=12 and index <= 27) or (index >= 32 and index <=39)):
        copyfile(dataset + dirs + '/' + files, trainset + dirs + '/' + files)
    else:
        copyfile(dataset + dirs + '/' + files, testset + dirs + '/' + files)
'''

#print('Folder ' + str(n) + ' completed! \n')







