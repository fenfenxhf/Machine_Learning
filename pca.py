from numpy import *
import matplotlib
import matplotlib.pyplot as plt

#数据录入与分析
def loadDataSet(filename,delim='\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

#主成分分析（PCA）：第一个坐标轴选择原始数据中方差最大的方向，第二个新的坐标轴选择与第一个坐标轴正交且具有最大方差的方向，迭代
def PCA(dataMat,topNfeat=99999999):
    meanVals = mean(dataMat,axis=0) #数据集按列求平均
    meanRemoved = dataMat - meanVals #去平均值
    covMat = cov(meanRemoved,rowvar=0) #计算协方差，cov(x,0)=cov(x)除数为n-1,cov(x,1)除数为n
    eigVals,eigVects = linalg.eig(mat(covMat)) #获得协方差矩阵的特征值和特征向量
    #print(eigVals)
    #print(eigVects)
    eigValInd = argsort(eigVals) #从小到大排序返回索引
    #print(eigValInd)
    eigValInd = eigValInd[:-(topNfeat+1):-1] #-1表示逆序,取前面N个索引
    #print(eigValInd)
    redEigVects = eigVects[:,eigValInd]  #保留最上面的N个特征向量，特征向量是以列向量存储
    #print(redEigVects)
    lowDDataMat = meanRemoved * redEigVects #原来的向量变成降维后的向量
    #print(lowDDataMat)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals #降维后的数据再次映射到原来空间中，用于与原始数据进行比较
    #print(reconMat)
    return lowDDataMat,reconMat

"""
#测试
dataMat = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch13\\testSet.txt')
lowDMat,reconMat = PCA(dataMat,1) # N = 1
print(shape(reconMat))

#将降维后的数据（又映射到原来的空间）和原始数据一起绘制出来
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90) #原始数据
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
plt.show()
"""

#示例：使用PCA对半导体制造数据降维
#将NaN（缺失值）替换成平均值函数,如果替换成0是下策，因为可能这条数据意思比较大，又因为基本上每条数据都有缺失值，也不能去除不完整样本
def replaceNaNWithMean():
    dataMat = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch13\\secom.data',' ')
    #print(dataMat[0])
    numFeat = shape(dataMat)[1]
    #print(dataMat[nonzero(~isnan(dataMat[:,0].A))[0],0])
    #print(~isnan(dataMat[:,0].A)) #打印Ture 或 Flase
    #print(nonzero(True)[0]) #返回索引
    for i in range(numFeat):
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0],i]) #第i列非NaN函数的均值
        #print(nonzero(~isnan(dataMat[:,i].A))[0])
        dataMat[nonzero(isnan(dataMat[:,i].A))[0],i] = meanVal
    return dataMat

"""
#测试
dataMat = replaceNaNWithMean()
meanVals = mean(dataMat,axis=0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved,rowvar=0)
eigVals,eigVects = linalg.eig(mat(covMat))
#print(eigVals)
"""

"""
#以下sklearn的实现
from sklearn.decomposition import PCA

dataMat = replaceNaNWithMean()

clf = PCA(n_components=6) #n_component是所降到的维数
clf.fit(dataMat)

newDataSet = clf.fit_transform(dataMat)
print(newDataSet)

print(clf.explained_variance_ratio_)
print(sum(clf.explained_variance_ratio_)) #0.9680647883717688
"""






