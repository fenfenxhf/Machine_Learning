from numpy import *
import numpy
import matplotlib.pyplot as plt
import random

#缺点：容易欠拟合，计算的代价较高
#Logistic回归梯度上升优化算法：首次接触到优化算法
def loadDataSet(): #由于需要进行距离计算，所以要求数据类型为数值型
    dataMat = [];labelMat = []
    fr = open(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch05\\testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) #x0=1 x1:lineArr[0],x2:lineArr[1]
        labelMat.append(int(lineArr[2])) #第三个作为分类标签
    return dataMat,labelMat

#定义sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

#算法核心
#梯度上升法：要找到某个函数的最大值，最好的方法就是沿着该函数的梯度方向探寻
def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn) #转化为numpy矩阵数据类型
    labelMat = mat(classLabels).transpose() #将标签矩阵转置成m*1的列矩阵
    m,n = shape(dataMatrix) #m*n的矩阵，m是数据集个数，n特征值个数
    alpha = 0.001 #向目标移动的步长，改变步长可以改变效率和准确率
    maxCycles = 500 #迭代次数
    weights = ones((n,1)) #初始化系数，n*1

    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights) #(mxn)*(nx1)>>m*1,矩阵中*代表真正意义上的矩阵乘法
        error = (labelMat - h) #计算真实类别与预测值之间的差值，接下来就是按照该差值的方向调整回归系数
        weights = weights + alpha * dataMatrix.transpose() * error #(n*m)*(m*1)>>n*1
    return weights #返回的weights是矩阵

"""
#测试
dataMat,labelMat = loadDataSet()
#print(shape(dataMat))
#print(shape(labelMat)) #打印结果(100,)
#print(shape(mat(labelMat).transpose())) #打印结果(100,1)
#print(labelMat) #打印结果[0, 1, 0, 0, 0, 1, 0, 1, 0,...]
weights = gradAscent(dataMat,labelMat)
print(weights) 
"""

#画出数据集和logistic回归最佳拟合直线函数
def plotBestFig(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat) #将矩阵数组化
    #print(dataArr[1,1])
    m = shape(dataMat)[0]
    xcord1 = [];ycord1 = [] #存放类别1和0的点
    xcord2 = [];ycord2 = []

    for i in range(m):
        if int(labelMat[i]) == 1: #如果类别标签为1
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2]) #dataArr第一列是x1,第0列是x0(==1)
        else:
            xcord2.append(dataArr[i, 1]);ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s') #s表示方块形,scatter表示画散点图
    ax.scatter(xcord2,ycord2,s=30,c='green') #默认圆形
    x1 = arange(-3.0,3.0,0.1) #步长为0.1，相当于特征值x1
    x2 = (-weights[0] - weights[1]*x1)/weights[2] #    #拟合曲线为0 = w0*x0+w1*x1+w2*x2, 故x2 = (-w0*x0-w1*x1)/w2, x0为1
    ax.plot(x1,x2) #plot表示画连续图
    plt.xlabel('x1');plt.ylabel('x2')
    plt.show()

"""
#测试
dataMat,labelMat = loadDataSet()
weights = gradAscent(dataMat,labelMat)
#print(weights)
#print(type(weights))
#plotBestFig(weights.getA()) ##getA()将矩阵转化成数组
"""


#随机梯度上升法：一次仅用一个样本更新回归系数
#因为在梯度上升法中每次更新回归系数都要遍历整个数据集，计算复杂度高
#随机梯度上升算法不需要使用矩阵，直接使用算法的numpy数组就可以了
def stocGradAscent0(dataMatrix,classLabels):
    #dataMatrix = array(dataMat)
    m,n = shape(dataMatrix)
    alpha = 0.8 #改变alpha值可以改变拟合的效果，目测0.8时比较好，但是还是不理想
    weights = ones(n) #产生1行n列的全1矩阵
    #print(dataMatrix[0]*weights)

    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights)) #第i个数据*W,是一行数（dataMatrix[0]*weights，
                                                # 结果是[ 1.       -0.017612 14.053064]），取sum之后就是一个数
        error = classLabels[i] - h #是一个数
        weights = weights + alpha * error * dataMatrix[i]  #一行n列
    return weights

"""
#测试
dataMat,labelMat = loadDataSet()
weights = stocGradAscent0(array(dataMat),labelMat)
print(weights)
#weightsmat = numpy.mat(weights)
#print(weightsmat[1].getA())
#print(type(weightsmat))
#plotBestFig(weights)
"""

#改进的随机梯度上升算法，收敛得更快
def stocGradAscent1(dataMatrix, classLabels, numIter=150): #默认迭代150次
    m, n = shape(dataMatrix)
    weights = ones(n)

    for j in range(numIter): #迭代次数
        dataIndex = list(range(m)) #将range(m)转化成列表形式，方便索引以及删除某个值，并且方便计算len(dataIndex)
        for i in range(m): #行数，也就是数据集的个数
            alpha = 4 / (1.0 + i + j) + 0.0001  # alpha随迭代次数不断改变缓解数据波动，1.非严格下降，2.不会到0
            # 随机选取样本更新系数weights,每次随机从列表中选取一个值，用过后删除它再进行下一次迭代
            randIndex = int(random.uniform(0, len(dataIndex)))  # 每次迭代改变dataIndex,而m是不变的，故不用unifor(0, m)
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del dataIndex[randIndex]
    return weights

"""
#测试
dataArr, labelMat = loadDataSet()
weights = stocGradAscent1(array(dataArr), labelMat)
print(weights)
plotBestFig(weights)
"""

#numpy数据类型不允许包含缺失值，可以用以下做法填补缺失值：使用可用特征的均值；使用特殊值如-1；忽略有缺失值的样本；
# 使用相似样本的均值来填补缺失值；使用另外的机器学习算法预测缺失值
#在逻辑回归可以用0替换所有的缺失值，因为weights = weights + alpha * error * dataMatrix[randIndex]，Xj为0是对应的系数不做更新
#同时sigmoid(0) = 0.5它对结果的预测不具有任何的倾向性

#示例：从疝气病症预测病马的死亡率
def classifyVect(inX,weights):
    prob = sigmoid(sum(inX*weights))#把测试集上的每个特征向量乘以最优化方法得到的系数再求和输入到sigmoid（）中
    if prob > 0.5:
        return 1.0
    else:
        return 0.0 ##sigmoid的值大于0.5判断为1，小于0.5判断为0

def colicTest():
    frTrain = open(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch05\\horseColicTraining.txt')
    frTest = open(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch05\\horseColicTest.txt')
    #解析训练集文本
    trainingSet = [];trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21): #文件中前21个表示特征值
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21])) #最后一个表示类别标签

    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount = 0.0;numTestVec = 0.0 #numTestVec测试文本行数
    #解析测试文本
    for line in frTest.readlines():
        numTestVec += 1.0 #每解析一行，本文行数就加1，最后就是测试集的个数
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVect(array(lineArr),trainWeights)) != int(currLine[21]): #用训练集得到的weights
            errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print("the error rate is : %f"%errorRate)
    return errorRate

#多次迭代训练，得出错误率的平均值
def multiTest():
    numTest = 10;errorSum = 0.0
    for k in range(numTest):
        errorSum += colicTest()
    print("afer %d iterations the average error rate is %f"%(numTest,errorSum/float(numTest)))


#测试
multiTest()



