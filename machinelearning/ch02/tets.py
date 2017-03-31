__author__ = 'Administrator'
import kNN
import numpy
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
'''
#机器学习K-临近算法，测试案例1
group,labels =kNN.createDataSet()
print(group,labels)
resulte=kNN.classify0([0.4,0.0],group,labels,3)
print(resulte)
'''

'''
s=1.0
for i in range(1,200):
    s=0.065
    if s>kNN.datingClassTest(i):
        s=kNN.datingClassTest(i)
        print('munber is %d,rate is %f' %(i,s))
'''

#testvector
#returnVect = numpy.zeros((1,1024))
#print(returnVect)
#testvector =kNN.img2vector('testDigits/0_0.txt')
#for i in range(1000):
 #   print(returnVect[0,i])
kNN.handwritingClassTest()

'''
hwLabels = []
trainingFileList = listdir('trainingDigits')           #load the training set
m = len(trainingFileList)
trainingMat = numpy.zeros((m,1024))
for i in range(3):
    fileNameStr = trainingFileList[i]
    fileStr = fileNameStr.split('.')[0]  #take off .txt
    classNumStr = int(fileStr.split('_')[0])
    hwLabels.append(classNumStr)
    ttt=kNN.img2vector('trainingDigits/%s' % fileNameStr)
    trainingMat[i,:] = kNN.img2vector('trainingDigits/%s' % fileNameStr)
'''
'''
testFileList = listdir('testDigits')        #iterate through the test set
errorCount = 0.0
mTest = len(testFileList)
'''