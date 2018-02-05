import os
from shutil import copyfile
import glob


dataset='bologna_dataset/'
trainset = 'bologna_train/'
testset = 'bologna_test/'

#files = glob.glob("*.txt")
#files.sort(key=os.path.getmtime)
#print("\n".join(files))


category = ''
n = 0

for dirs in os.listdir(dataset):
    n+=1
    #category = ''
    #length = 0

    os.mkdir(trainset + dirs)
    os.mkdir(testset + dirs)

    #for files in os.listdir(dataset + dirs):
        #length+=1

    #train_length = (length/100) * 80
    #test_length = (length/100) * 20

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



    #category += dirs + '___Tot: ' + str(int(length))+ '---> ' + ' Train: '+str(int(train_length))+ ' Test: '+str(int(test_length)) + '\n'
    #print(category)

    print('Folder ' + str(n) + ' completed! \n')







