from numpy import *
import matplotlib.pyplot as plt
from time import sleep

#标准回归函数和数据导入
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1 #特征数
    dataMat = [];labelMat = []
    fr = open(filename)

    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#w = ((X.T*X).I)*X.T*y
def standRegres(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T #m*n;m*1
    xTx = xMat.T * xMat #(n*m) * (m*n) >> n*n
    if linalg.det(xTx) == 0.0: #如果不可逆。因为要求逆，所以要先判断是否可逆
        print("This matrix is singular,cannot do inverse")
        return
    ws = xTx.I * xMat.T * yMat #n*n n*m m*1  >>> n*1
    return ws

"""
#测试
xArr,yArr = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch08\\ex0.txt')
#print(xArr)
#print(yArr)
ws = standRegres(xArr,yArr) # n*1
print(ws)
xMat = mat(xArr)
yMat = mat(yArr) #1*m
yHat = xMat * ws #m*n n*1 >>> m*1
print(corrcoef(yHat.T,yMat)) #计算预测值和真实值之间的相关系数(匹配程度)，要保证两个向量都是行向量
#print(yPredMat.T)

#画原始数据图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0]) #flatten()方法能将matrix的元素变成一维的
#print(xMat[:,1]) #m*1矩阵
#print(xMat[:,1].flatten())
#print(xMat[:,1].T) #和上一行等价效果
#plt.show()

#画拟合直线图
xCopy = xMat.copy()
#print(xCopy)
xCopy.sort(0) #0是按列遍历，1是按行遍历
print(xCopy) #不用重新赋值，排序后的自动覆盖xCopy
#print(xCopy[:,1].sort(0))#为了避免直线上的数据点次序混乱，绘图时将出现问题，我们要将点按照升序排列
yHat = xCopy * ws
#print(yPredMat)
ax.plot(xCopy[:,1],yHat,c='red')
plt.show() #不能同时出现两个plt.show,所以要先把上面的注释了
"""

#局部加权线性回归函数（LWLR）：给在待测点附近的每一点赋予一定的权重（使用高斯核）
# 因为求的是具有最小均方误差的无偏估计，所以可能会出现一些欠拟合的情况，降低欠拟合问题，加入一些偏差，从而降低预测的均方误差
##w = ((X.T*W*X).I)*X.T*W*y 其中W是一个矩阵，用来给每个数据点赋予权重
#W[i,i] = exp(|x[i,:]-x| / -2*k**2)
def lwlr(testPoint,xArr,yArr,k=1.0): #testPoint就是一个测试数据点，k是高斯核的一个参数
    xMat = mat(xArr);yMat = mat(yArr).T #m*n m*1
    m = shape(xMat)[0]
    weights = mat(eye(m)) #创建对角矩阵

    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T / (-2.0*k**2))

    xTx = xMat.T * weights * xMat #n*m m*m m*n >> n*n
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = xTx.I * xMat.T * weights * yMat #n*n n*m m*m m*1 >> n*1
    return testPoint * ws #返回预测的值

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m) #m*1
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

"""
#测试
xArr,yArr = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch08\\ex0.txt')
yHat = lwlrTest(xArr,xArr,yArr,0.01) #0.003时噪声太多，容易过拟合,0.01时候比较反映数据的真实意义
#print(yHat)
xMat = mat(xArr)
xMat_copy = xMat.copy()
xMat_copy.sort(0)
srtIndex = xMat[:,1].argsort(0) #从小到大排序返回索引，0表示按列
#xSort = xMat[srtIndex][:,0,:]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xMat_copy[:,1],yHat[srtIndex],c='red')
ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=10)
plt.show()
"""

#示例：预测鲍鱼的年龄
def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum() #分析误差大小

"""
#测试
abX,abY = loadDataSet(r"E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch08\\abalone.txt")
yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
error01 = rssError(abY[0:99],yHat01.T) #较小的核形成较低的误差，但是容易过拟合
error1 = rssError(abY[0:99],yHat1.T)
error10 = rssError(abY[0:99],yHat10.T)
print(error01) #56.821685348787405
print(error1) #429.8905618700201
print(error10)#549.1181708827194
#试试新数据上的表现
yHat01c = lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
yHat1c = lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
yHat10c = lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
error01c = rssError(abY[100:199],yHat01.T) #较小的核形成较低的误差，但是容易过拟合
error1c = rssError(abY[100:199],yHat1.T)
error10c = rssError(abY[100:199],yHat10.T)
print(error01c) #2824.1796760076863
print(error1c) #2168.439135582531
print(error10c)#2105.4837116482568
"""

#岭回归:缩减系数，当数据的特征比样本点还多时，则x不是满秩矩阵，则计算(X.T*X).I会出错
#缩减参数，则模型的复杂度降低，出现高偏差；若模型的复杂度越高，说明拟合训练集数据越好，重新给数据拟合可能就不好了，
# 两次拟合所得系数之间的差异就是模型方差大小的反映
##w = ((X.T*X + lambdaI).I)*X.T*y  lambda来限制所有w的和，引入该惩罚项，能够减少不重要的参数
def ridgeRegres(xMat,yMat,lam=0.2):#lambda是python中的关键字，所以用lam
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = denom.I * xMat.T * yMat
    return ws

#在30不同的lam下调用ridgeRegres()函数
def ridgeTest(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    yMean = mean(yMat,0) #求每一类的平均值，0就是按列向下索引
    #print(yMean)
    yMat = yMat - yMean #需要对特征值做标准化的处理，都减去各自特征的平均值
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)#求方差
    #print(xVar)
    xMat = (xMat - xMeans) / xVar #对数据做归一化
    numTestPts = 30 #30个不同的lam 给定不同的lam,得到不同的ws
    wMat = zeros((numTestPts,shape(xMat)[1])) #30*n

    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat

"""
#测试
abX,abY = loadDataSet(r"E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch08\\abalone.txt")
wMat = ridgeTest(abX,abY)
print(wMat)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(wMat) #按列数据来画，应该有shape(xMat)[1]这么多条曲线
plt.show()
"""

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

#前向逐步回归:贪心算法，每一步都尽可能减少误差，一开始所有的权重都为1，然后每一步所做的决策是对某个权重增加或减少一个很小的值
#当一个模型构建好之后，可以用这个算法找出重要的特征，就有可能及时停止对不重要特征的收集
def stageWise(xArr,yArr,eps=0.04,numIt=100):
    xMat = mat(xArr);yMat = mat(yArr).T
    yMean = mean(yMat,0) #标准化数据：0均值
    yMat = yMat - yMean
    xMat = regularize(xMat) #标准化数据：单位方差
    m,n = shape(xMat)
    returnMat = zeros((numIt,n)) #存储ws
    ws = zeros((n,1));wsTest = ws.copy();wsMax = ws.copy()

    for i in range(numIt):#每一次迭代
        #print(ws.T)
        lowestError = inf #正无穷

        for j in range(n):#每一个特征

            for sign in [-1,1]:#增大或减小对应的w,取值为1或者-1
                wsTest = ws.copy()
                wsTest[j] += eps*sign #增大或减小
                yTest = xMat * wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

"""
#测试
xArr,yArr = loadDataSet(r"E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch08\\abalone.txt")
returnMat = stageWise(xArr,yArr,0.001,4000)
print(returnMat)
"""







