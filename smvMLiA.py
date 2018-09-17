#SMO（序列最小优化）算法(目标找出一系列alpha和b)中的辅助函数
#SVM是很好的现成的分类器，现成是指分类器不加修改就可以直接使用（仅使用于处理二类问题）
#我们希望找到离分割超平面最近的点，确保它们离分割超平面尽可能的远
# 支持向量就是离分割超平面最近的那些点，所以要最大化支持向量到分割面的距离
import random
from numpy import *
from os import listdir

#解析文本并得到每行的类标签和整个行矩阵
def loadDataSet(filename):
    dataMat = [];labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#随机选择两个参数i,j(alpha的下标)
#每次选择两个alpha进行优化处理，因为有约束条件sum(alpha(i) * label(i)) = 0,i是下标。如果只改变一个alpha则约束条件失效
def selectJrand(i,m): #i是alpha的下标，m是alpha的总的个数
    j = i
    while j==i:
        j = int(random.uniform(0,m)) #循环执行，保证j != i，也就是选择了两个不同的alpha
    return j

#辅助函数，调整大于H或者小于L的alpha，因为alpha有约束条件C >= alpha >= 0，C是松弛变量允许数据点可以处于分割面的错误一侧
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

"""
#测试
dataMat,labelMat = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch06\\testSet.txt')
print(labelMat) #标签用-1，+1。因为间隔等于label * (w.T * x +b),如果数据点处于正方向（+1类）并且距离分割超平面很远的位置
#(w.T * x +b)是一个很大的正数；如果数据点处于负方向（-1类）并且距离分割超平面很远的位置，(w.T * x +b)是很大的负数，
#由于标签是-1，所以label * (w.T * x +b)仍然是个很大的正数
"""

#简化版的SMO算法：处理小规模数据集。简化版是在数据集上遍历每一个alpha，然后在剩下的alpha集合中随机选择另一个alpha
#理解计算过程可看统计学习方法（李航）的支持向量机部分
def smoSimple(dataMat,classLabels,C,toler,maxIter):#数据集，分类标签，控制参数（惩罚参数），容错率，最大迭代次数
    dataMatrix = mat(dataMat) #m*n
    labelMat = mat(classLabels).transpose() #转化成矩阵形式方便后面计算,m*1
    b = 0 #初始化b
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m,1))) #有m个alpha,初始化alphas, m*1 的全0
    iter = 0 #存储的是没有任何alpha改变的情况下遍历数据集的次数

    while iter < maxIter: #任何alpha改变的情况下遍历数据集的次数达到最大值，则退出
        alphaPairsChanged = 0 #记录alpha是否已经优化，如果不需要优化，则iter要加1

        for i in range(m): #内循环：对数据集的每一个数据向量
            # w = sum(alpha[i] * y[i] * x[i])
            #分离超平面：sum(alpha[i] * y[i] * (x * x[i])) + b
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b # 一个数
            #上式表示决策函数只依赖于输入x和训练样本输入的内积（公式见统计学习方法（李航）P106  7.30）
            #在矩阵中，multiply表示矩阵元素对应相乘，*表示矩阵乘法，W.T*Xi+b,预测的类别
            Ei = fXi - float(labelMat[i]) #计算预测和实际的误差，如果误差很大则需要优化对应的alpha
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #不管树正间隔还是负间隔都要测试，同时检查alpha使其不能等于0或者C
                j = selectJrand(i,m) #随机选择第二个alpha
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])

                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()#为alphaIold alphaJold分配新的内存
                #将alpha进行剪辑，具体公式见统计学习方法（李航）P126
                if labelMat[i] != labelMat[j]:#y1 y2异号
                    L = max(0,alphas[j] - alphas[i]) # L>=0
                    H = min(C,C + alphas[j] - alphas[i]) # H<=C
                else: #y1 y2同号
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,C + alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue #在边界无需优化,因而不能够增大或者减少，也就不值得进行优化

                # eta是alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T \
                      - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T  #？
                if eta >= 0:
                    print("eta>=0")
                    continue

                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j],H,L) #对alpha[j]进行剪辑

                if abs(alphas[j] - alphaJold) < 0.00001:#alphaj如果没有调整
                    print("j not moving enough")
                    continue
                #修改i，修改量与j相同，但方向相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:]*dataMatrix[i,:].T - \
                    labelMat[j]*(alphas[j] - alphaJold) * dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMat[j]*(alphas[j] - alphaJold) * dataMatrix[j,:]*dataMatrix[j,:].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('iter: %d i: %d, pairs changed %d' %(iter, i, alphaPairsChanged))

        # 如果没有更新，那么继续迭代；如果有更新，那么迭代次数归0，继续优化
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0 #只要有修改的alpha对，就将iter置0，就是迭代次数重新计算
        print('iteration number: %d' % iter)
    return b,alphas #alpha不等于0的为支持向量

"""
#执行速度真的好慢
#测试
dataMat,labelMat = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch06\\testSet.txt')
b,alphas = smoSimple(dataMat,labelMat,0.6,0.001,40)
print(b)
print(alphas)
print(shape(alphas[alphas>0]))
for i in range(100):
    if alphas[i]>0.0:
        print(dataMat[i],labelMat[i])
"""

#利用完整版Platt SMO算法加速优化
#完整版Platt SMO的支持函数
class optStruct(object): #定义一个类
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):#数据集，数据集标签，松弛变量，容错率，kTup是一个包含核函数信息的元组
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1))) #定义并初始化alphas
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #误差缓存，第一位为有效位，第二位为误差,m*2
        #因为alpha[j]的选择是依赖|Ei-Ej|，为了节约计算时间，将所有的Ei保存在一个矩阵中
        #添加一个包含核函数信息的二元组
        self.K = mat(zeros((self.m,self.m))) #m*m
        for i in range(self.m): #数据集中的每一条数据
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup) #K的第i列存储第i条数据核函数的值


#在复杂数据上运用核函数:将内积替换成核函数的方式被称为核技巧
#核转换函数
def kernelTrans(X,A,kTup): #数据集，某一行数据X[i,:]，核函数信息
    m,n = shape(X)
    K = mat(zeros((m,1))) #m*1

    if kTup[0] == 'lin':#lin表示线性核函数
        K = X * A.T  # x * x[i]
    elif kTup[0] == 'rbf': #rbf表示径向基核函数
        for j in range(m):
            deltaRow = X[j,:] - A #第j行和第i行每个元素的差值
            K[j] = deltaRow * deltaRow.T #一个数
        K = exp(K / (-1*kTup[1]**2)) #kTup包含核函数信息二元组，其中kTup[1]是用户定义的用于确定到达率或者说函数跌落到0的速度参数
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    #print(K)
    return K #m*1


#计算误差
def calcEk(oS,k):
    #fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b

    #核函数中改变fXk
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)

    Ek = fXk - float(oS.labelMat[k])
    return Ek

#选择j,内循环中的启发方法:选择第二个变量
def selectJ(i,oS,Ei):
    maxK = -1;maxDeltaE = 0;Ej = 0
    oS.eCache[i] = [1,Ei] #根据Ei更新误差缓存，第一位为有效位，第二位为误差
    validEcacheList = nonzero(oS.eCache[:,0].A)[0] #返回误差不为0的数据的索引值

    if len(validEcacheList) > 1:#如果有误差不为0
        for k in validEcacheList:
            if k == i:
                continue #不计算Ei的值，节约算法时间
            Ek = calcEk(oS,k)
            deltaE = abs(Ei - Ek) #选择具有最大步长的j，alpha[j]的选择是依赖于|E1-E2|,为了加快计算速度，简单的做法
                                  #就是使其对应的|E1-E2|最大
            if deltaE > maxDeltaE:
                maxK = k;maxDeltaE = deltaE;Ej = Ek
        return maxK,Ej
    else: #如果所有的误差均为0
        j = selectJrand(i,oS.m) #随意选择一个j(不等于i)
        Ej = calcEk(oS,j)
    return j,Ej

#更新误差缓存
def updateEk(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]

#完整Platt SMO算法中的优化例程
def innerL(i,oS):
    Ei = calcEk(oS,i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i,oS,Ei) #第二个alpha选择用启发式方法
        alphaIold = oS.alphas[i].copy();alphaJold = oS.alphas[j].copy()

        #将alpha进行剪辑，具体公式见统计学习方法（李航）P126
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: #在边界上
            print("L==H")
            return 0

        #eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        #使用核函数要修改eta
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]

        if eta >= 0: #alpha_j的最优修改量
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L) #修剪alpha_j

        updateEk(oS,j) #因为alpha[j]已经修改，更新误差缓存

        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0

        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i) #因为alpha[i]已经修改，更新误差缓存

        #更新b
        #b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * \
             #(oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        #b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * \
             #(oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T

        #核函数中改变b1 b2
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * \
             (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * \
             (oS.alphas[j] - alphaJold) * oS.K[j, j]


        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1 #alpha[i] alpha[j]有改变则返回1
    else:
        return 0 #对应的alpha没有变化，则返回0

#完整版Platt SMO的外循环代码:选择第一个变量
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)  #建立一个数据结构来保存所有的重要值
    iter = 0
    entireSet = True;alphaPairsChanged = 0

    while (iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        ##遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
        alphaPairsChanged = 0
        if entireSet: #遍历整个数据集
            for i in range(oS.m): #遍历所有的值
                alphaPairsChanged += innerL(i,oS) #alpha的优化次数
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1 #这里是一次迭代定义为一次循环过程，而不管该循环具体做了什么事情
            #但是在smoSimple()中没有任何alpha发生改变时会将整个数据集的一次遍历记成一次迭代，
            #如果在优化过程中存在波动就会停止，因此这里的做法优于smoSimple()
        else:#遍历非边界值，其中0 < alpha < C 是在间隔边界上的支持向量
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1

        if entireSet: #如果遍历了整个数据集，则遍历非边界值
            entireSet = False
        elif alphaPairsChanged == 0: #如果alpha[i]没有任何改变，则遍历整个数据集
            entireSet = True
        print("迭代次数: %d" % iter)
    return oS.b,oS.alphas  #非零alpha对应的就是支持向量

"""
#测试
dataMat,labelMat = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch06\\testSet.txt')
b,alphas = smoP(dataMat,labelMat,0.6,0.001,40)
print(b)
print(alphas)
"""

#如何用计算出的b和alpha去计算W，并利用他们进行分类
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr);labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1)) #n*1

    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

"""
#测试
dataArr,labelMat = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch06\\testSet.txt')
b,alphas = smoP(dataArr,labelMat,0.3,0.001,70)
ws = calcWs(alphas,dataArr,labelMat)
print(ws)
dataMat = mat(dataArr)
m = shape(dataMat)[0]
labelP = [];errSum = 0
for i in range(m):
    label = sign(dataMat[i]*mat(ws)+b)
    labelP.extend(label.A.tolist()[0])#矩阵转数组转列表取第一位，还是列表
print(labelMat)
print(labelP)
for i in range(m):
    if labelP[i] != labelMat[i]:
        errSum += 1
print(errSum/m) 
"""

#利用核函数进行分类的径向基函数
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode'
                                   r'\\machinelearninginaction\\Ch06\\testSetRBF.txt')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0] #返回非零alpha的索引，也就是支持向量的索引,一般用nonzero后面都有[0]
    sVs = dataMat[svInd] #支持向量
    labelSV = labelMat[svInd] #支持向量对应的标签
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(dataMat)
    errorCount = 0
    #如何利用核函数进行分类
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1)) #用支持向量和数据集每一行 返回K
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

    dataArr, labelArr = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode'
                                    r'\\machinelearninginaction\\Ch06\\testSetRBF2.txt') #与上面的测试集不同
    errorCount = 0
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)

    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1)) #用支持向量和数据集每一行
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))

"""
#测试
#testRbf()
"""

#手写识别问题回顾
#将图像（32*32）转化为1*1024的向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32): #第i行
        lineStr = fr.readline()
        for j in range(32): #第j列
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList) #文件个数，也就是后面数据集的个数
    trainingMat = zeros((m,1024)) #提取数据

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1) #数字9的标签为-1
        else:
            hwLabels.append(1) #其他数字的标签为1，因为支持向量机是个二分类器，这里只区分是不是9
        trainingMat[i,:] = img2vector(r"%s\\%s"%(dirName,fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr = loadImages(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch02\\digits\\trainingDigits')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,kTup)
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]  # 返回非零alpha的索引，也就是支持向量的索引
    sVs = dataMat[svInd]  # 支持向量
    labelSV = labelMat[svInd]  # 支持向量对应的标签
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    # 如何利用核函数进行分类
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)  # 用支持向量和数据集每一行而不是用整个数据集，节约计算时间
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

    dataArr, labelArr = loadImages(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch02\\digits\\testDigits')
    errorCount = 0
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)

    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :],kTup)  # 用支持向量和数据集每一行
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))

"""
#测试
testDigits(('rbf',20))
"""






