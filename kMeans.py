from numpy import *
#import random
#import urllib
#import json
#from time import sleep
import matplotlib
import matplotlib.pyplot as plt

#K-均值聚类的支持函数:聚类是一种无监督的学习(无监督学习没有训练过程)，将相似的对象归为同一个簇，K是指K个簇
#导入并解析数据
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        fltLine3 = list(fltLine) #python3的用法
        dataMat.append(fltLine3)
    return dataMat

#计算误差，这里用的是两个向量的欧式距离
def distEclud(vecA,vecB):
    return sum(power(vecA-vecB,2)) ** 0.5

#构建簇质心:随机选择k个簇质心
def randCent(dataSet,k):#k表示k个簇
    n = shape(dataSet)[1]
    centroid = mat(zeros((k,n))) #存储随机质心 k*n
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroid[:,j] = minJ + rangeJ * random.rand(k,1) #随机点保证在数据的边界之内
        #random.rand(i,j) #随机生成一个i*j的矩阵，其中每个元素的值都在0-1之间
    return centroid

"""
#测试
datMat = mat(loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch10\\testSet.txt'))
#print(datMat[0]) #矩阵的第一个数据
dist = distEclud(datMat[0],datMat[1]) #第一个数据和第二个数据之间的距离
print(dist)
centroid = randCent(datMat,2)
print(centroid)
#print(random.random())
"""

#k-均值聚类算法
def kMean(dataSet,k,distMeas=distEclud,createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2))) #第一列是第几簇，第二列记录每个数据点到簇质心的距离平方 m*2
    centroids = createCent(dataSet,k) #随机获得K个质心
    clusterChange = True
    while clusterChange:#当任意一个点的簇分配发生改变时
        clusterChange = False

        for i in range(m): #对每个数据点
            minDist = inf;minIndex = -1

            for j in range(k):#对每一个簇质心
                distJI = distMeas(centroids[j,:],dataSet[i,:]) #计算该数据点到每一个质心的距离
                if distJI < minDist: #如果该数据到j质心的距离变小了
                    minDist = distJI;minIndex = j #那么就将该数据点分配到第j个质心里面
            if clusterAssment[i,0] != minIndex: #如果第I个数据的分类发生变化
                clusterChange = True
            clusterAssment[i,:] = minIndex,minDist**2 #更新第i个数据存储信息（第几簇和到这个簇质心的距离）
        #print(centroids)

        for cent in range(k): #求质心
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]] #取出某一簇的所有数据，返回索引第几条数据
            centroids[cent,:] = mean(ptsInClust,axis=0) #按列求平均
    return centroids,clusterAssment

"""
#测试
datMat = mat(loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch10\\testSet.txt'))
mycentroids,clustAssing = kMean(datMat,4)
print(mycentroids)
print(clustAssing)
"""

#二分K-均值聚类算法
def biKmeans(dataSet,k,distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet,axis=0).tolist()[0] #将所有点看成了一个簇，按列计算均值，即所有数据的质心
    centList = [centroid0]
    #print(centList)
    for j in range(m): #对每一条数据
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:]) ** 2 #计算每个点到均值之间的距离

    while (len(centList) < k): #当簇的数目小于k时
        lowestSSE = inf

        for i in range(len(centList)): #对每一个簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:] #把属于每个簇的元素取出来
            centroidMat,splitClustAss = kMean(ptsInCurrCluster,2,distMeas) #k=2二分法划分
            sseSplit = sum(splitClustAss[:,1]) #二分之后的误差
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1]) #没有被分割簇的误差
            #print('sseSplit,and notSplit: ',sseSplit,sseNotSplit)

            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat #存储每一个簇的均值，即质心
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseNotSplit + sseSplit

        #更新簇的结果
        ##因为是二分，分的簇只有两个（0,1），其中1簇的簇的序号改为len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        ##因为是二分，分的簇只有两个（0,1），其中0簇的簇的序号改为bestCentToSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        #print('the bestCentToSplit is: ',bestCentToSplit)
        #print('the len of bestClustAss is: ',(len(bestClustAss)))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] #将一个质心更新成两个质心
        centList.append(bestNewCents[1,:].tolist()[0])
        #更新clusterAssment列表中参与2 - 均值聚类数据点变化后的分类编号，及数据该类的误差平方
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return mat(centList),clusterAssment

"""
#测试
datMat = mat(loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch10\\testSet2.txt'))
centList,myNewAssments = biKmeans(datMat,3)
#print(centList)
"""

#示例：对地理位置进行聚类
#两点球面距离计算
def distSLC(vecA,vecB):
    a = sin(vecA[0,1] * pi / 180) * sin(vecB[0,1] * pi / 180)
    b = cos(vecA[0,1] * pi / 180) * cos(vecB[0,1] * pi / 180) * cos(pi * (vecB[0,0] - vecA[0,0]) / 180)
    return arccos(a + b) * 6371.0

def clusterClubs(numClust=5):
    datList = []
    fr = open(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch10\\places.txt')
    for line in fr.readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[-1]),float(lineArr[-2])]) #获得经纬度
    #print(datList)
    datMat = mat(datList)
    myCentroids, clusterAssing = biKmeans(datMat,numClust,distMeas=distSLC)
    #print(myCentroids)
    #print(clusterAssing)

    fig = plt.figure()
    rect = [0.1,0.1,0.8,0.8] #从（0.1,0.1）开始画，横纵坐标都是0.8
    scatterMarkers = ['s', 'o', '^', '8', 'p','d', 'v', 'h', '>', '<']
    axprops = dict(xticks=[],yticks=[])
    ax0 = fig.add_axes(rect,label='ax0',**axprops)
    imgP = plt.imread(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch10\\Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect,label='ax1',frameon=False)

    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clusterAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]  # 使用索引来选择标记形状
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0],myCentroids[:,1].flatten().A[0], marker = '+', s = 300)
    plt.show()

"""
#测试
clusterClubs()
"""

"""
#以下用sklean来实现对地理坐标的聚类
from sklearn.cluster import KMeans

#数据处理
def loadData(filename):
    datList = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[-1]),float(lineArr[-2])]) #获得经纬度
    #print(datList)
    datMat = mat(datList)
    return datMat

X = loadData(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch10\\places.txt') #无标签

clf = KMeans(n_clusters=5)
clf.fit(X)

print(clf.labels_) #打印每一个点的聚类类别，这里是0,1,2,3,4  一共五个簇
print(clf.cluster_centers_) #聚类中心
print(clf.predict([[-125.00,45.09]]))
print(clf.score(X))
"""







