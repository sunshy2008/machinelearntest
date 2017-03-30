__author__ = 'Administrator'
import kNN
import numpy
import matplotlib
import matplotlib.pyplot as plt
'''
#机器学习K-临近算法，测试案例1
group,labels =kNN.createDataSet()
print(group,labels)
resulte=kNN.classify0([0.4,0.0],group,labels,3)
print(resulte)
'''
s=1.0
for i in range(1,200):
    s=0.065
    if s>kNN.datingClassTest(i):
        s=kNN.datingClassTest(i)
        print('munber is %d,rate is %f' %(i,s))
