import kNN
import numpy
import matplotlib
import matplotlib.pyplot as plt


datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt')
datingDataMat,datingLabels

fig = plt.figure()
ax =fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,0],15.0*numpy.array(datingLabels),15.0*numpy.array(datingLabels))
plt.show()