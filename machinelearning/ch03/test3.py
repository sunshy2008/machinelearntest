__author__ = 'Administrator'
from ch03 import trees
MyDat,Labels = trees.createDataSet()
#sang =trees.calcShannonEnt(MyDat)
bestFeature=trees.chooseBestFeatureToSplit(MyDat)
print(bestFeature)
