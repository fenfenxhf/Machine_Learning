from numpy import *
import matplotlib.pyplot as plt


##基于单层决策树构建弱分类器
def loadSimpleData():
    dataMat = matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


# 单层决策树生成函数：基于单个特征做决策，也就是个树桩
#下面函数是通过阈值比较对数据进行分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # 数据集，第dimen+1个特征，阈值，两种阈值过滤模式
    retArray = ones((shape(dataMatrix)[0], 1))  # 预测出数据集标签向量,初始化m*1的全1矩阵
    if threshIneq == 'lt':  # 阈值过滤方式1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
        # 某一列的数值小于等于阈值，则赋值为-1.0（阈值一边赋值1，另一边赋值-1）
    else:  # 阈值过滤方式2
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

#找到最佳单层决策树
def buildStump(dataArr, classLabels, D):  # 数据集，数据集标签，基于数据的权重向量(m*1)
    dataMatrix = mat(dataArr)
    labelsMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0 # numSteps用于在特征的所有可能值上面遍历
    bestStump = {} #存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestClassEst = mat(zeros((m, 1))) # bestClassEst是预测的数据集标签向量
    minError = inf  # 正无穷，最小的误差设定为正无穷，然后重新赋值

    for i in range(n):  # 第一层循环：对数据集的每一个特征
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 步长

        for j in range(-1, int(numSteps) + 1):  # 第二层循环：每一个步长 [-1,10]
            #主要计算阈值，以使得特征的所有可能只都遍历到

            for inequal in ['lt', 'gt']:  # 第三层循环：对每一个不等号，这是两种不同的过滤模式
                threshVal = (rangeMin + float(j) * stepSize)  # 阈值计算
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 预测每一个数据的标签，得到一个分类器
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelsMat] = 0  # 将错误向量中正确分类的位置置0,m*1
                weightedError = D.T * errArr  # 这是一个数，计算分类误差率，见统计学习方法（李航）P139 8.8
                # print("split:dim %d,thresh %f,thresh inequal:%s,the weighted error is %f"%(i,threshVal,inequal,weightedError))
                # 存储信息
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()  # 重新分配内存
                    bestStump['dim'] = i #得到第i个特征，进行后面的决策
                    bestStump['thresh'] = threshVal #阈值
                    bestStump['ineq'] = inequal #过滤模式
    return bestStump, minError, bestClassEst


"""
#测试
dataMat,classLabels = loadSimpleData()
D = mat(ones((5,1))/5)
bestStump,minError,bestClassEst = buildStump(dataMat,classLabels,D)
print(bestStump)
print(minError)
print(bestClassEst)
"""


# 基于单层决策树得到多个弱分类器
def adaBoostTrainDS(dataArr, classLabels, numIt):  # 数据集，分类标签，迭代次数
    weakClassArr = []  # 存储得到的单层决策树的信息
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # 一开始每个数据的权重相等(1/m)
    aggClassEst = mat(zeros((m, 1)))  # 初始化一开始的所有数据的标签，记录每个数据点的类别估计累计值

    for i in range(numIt):  # 每一次迭代
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #返回最佳单层决策树的信息；最小的误差；；在此分类下的所有数据的标签向量
        #print("D: ", D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        # 更新每个分类器的权值，max(error,1e-16)确保在没有发生错误是出现除零溢出
        bestStump['alpha'] = alpha #alpha是分类器的权值
        weakClassArr.append(bestStump)
        #print("classEst: ", classEst.T)

        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # 更新D，这是exp的指数部分
        D = multiply(D, exp(expon))
        D = D / D.sum()

        aggClassEst += alpha * classEst  # 累加每一个预测值和该分类器权重的乘积
        #print("aggClassEst: ", aggClassEst.T)
        aggError = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))  # 将预测错的位置置1，正确的置0
        # 这一行可以用以下两行被注释的代码代替， # sign(t)函数：在t>=0是，sign(t)=1;在t<0是，sign(t)=-1 符号函数
        # aggError = mat(zeros((m,1)))
        # aggError[sign(aggClassEst) != mat(classLabels).T] = 1
        errorRate = aggError.sum() / m  # 错误比例
        #print("total error rate: ", errorRate, '\n')
        if errorRate == 0.0:
            break  # 如果错误比例已经为0则终止循环
    return weakClassArr,aggClassEst


"""
#测试
dataMat,classLabels = loadSimpleData()
weakClassArr,aggClassEst = adaBoostTrainDS(dataMat,classLabels,9)
print(weakClassArr,aggClassEst)
"""


# 测试算法：基于AdaBoost的分类
# 分类函数
def adaClassify(datToClass,classifierArr): #待分类的样例；多个弱分类器组成的数组
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))

    for i in range(len(classifierArr)):  # 对每一个分类器进行遍历预测
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        #print(aggClassEst)
    return sign(aggClassEst) #符号函数

"""
# 测试
dataMat,classLabels = loadSimpleData()
weakClassArr = adaBoostTrainDS(dataMat,classLabels,30)
print(weakClassArr)
classifiedLabels = adaClassify([[0,0],[5,5]],weakClassArr)
#print(classifiedLabels)
"""

#曾经用Logistic回归预测疝病马是否存活，现在利用单层决策树和adaBoost预测
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = [];labelMat = []
    fr = open(filename)

    for line in fr.readlines():
        lineArr = []
        currLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(currLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))
    return dataMat,labelMat

"""
#测试
dataArr,labelArr = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch07\\horseColicTraining2.txt')
classifierArray = adaBoostTrainDS(dataArr,labelArr,30) #得到若分类器
testArr,testLabelArr = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch07\\horseColicTest2.txt')
m = shape(testArr)[0]
prediction = adaClassify(testArr,classifierArray)
errArr = mat(ones((m,1)))
sumerr = errArr[prediction != mat(testLabelArr).T].sum()
errRate = sumerr / m
print(errRate)
"""


#ROC曲线的绘制及AUC（曲线下的面积）计算函数，计算不同的阈值情况下，假阳率（X轴），真阳率（Y轴），的曲线
def plotROC(predStrengths,classLabels):#predStrengths是所有分类器对应预测值的和，也就是aggClassEst
    cur = (1.0,1.0) #浮点数二元组，初始值是（1.0,1.0），该元组保留的是绘制光标的位置
    ySum = 0.0 #用于计算AUC的值
    numPosClas = sum(array(classLabels)==1.0) #计算真实数据中标签为1的个数
    yStep = 1 / float(numPosClas) #x步长
    xStep = 1 / float(len(classLabels)-numPosClas) #y步长
    sortedIndicies = predStrengths.argsort() #从小到大排序

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    for index in sortedIndicies.tolist()[0]:#.tolist将numpy的元组或矩阵改成列表形式
        if classLabels[index] == 1.0:#数据集标签中为1时，改变Y轴坐标
            delX = 0;delY = yStep
        else:#数据集标签中为1时，改变X轴坐标
            delX = xStep;delY = 0
            ySum += cur[1] #计算每个数据y的高度，因为小矩形的宽度是xStep，AUC=xStep*ySum
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate');plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)

"""
#测试
dataArr,labelArr = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch07\\horseColicTraining2.txt')
classifierArray,aggClaaEst = adaBoostTrainDS(dataArr,labelArr,10)
plotROC(aggClaaEst.T,labelArr)
"""


##基于单层决策树构建弱分类器
def loadSimpleData():
    dataMat = matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


# 单层决策树生成函数
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # 数据集，第dimen+1个特征，阈值，两种阈值过滤模式
    retArray = ones((shape(dataMatrix)[0], 1))  # 预测出数据集标签向量
    if threshIneq == 'lt':  # 阈值过滤方式1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0  # 某一列的数值小于等于阈值，则赋值为-1.0（阈值一边赋值1，另一边赋值-1）
    else:  # 阈值过滤方式2
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):  # 数据集，数据集标签，基于数据的权重向量
    dataMatrix = mat(dataArr);
    labelsMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0;
    bestStump = {};
    bestClassEst = mat(zeros((m, 1)))
    # numSteps用于在特征的所有值上面遍历，bestStump存储给定权重D时所得的最佳单层决策树的相关信息，bestClassEst是预测的数据集标签向量
    minError = inf  # 正无穷，最小的误差设定为正无穷，然后重新赋值

    for i in range(n):  # 第一层循环：对数据集的每一个特征
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 步长

        for j in range(-1, int(numSteps) + 1):  # 第二层循环：每一个步长

            for inequal in ['lt', 'gt']:  # 第三层循环：对每一个不等号，这是两种不同的过滤模式
                threshVal = (rangeMin + float(j) * stepSize)  # 阈值计算
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 预测每一个数据的标签
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelsMat] = 0  # 将错误向量中正确分类的位置置0
                weightedError = D.T * errArr  # 这是一个数，算出错误的权重
                # print("split:dim %d,thresh %f,thresh inequal:%s,the weighted error is %f"%(i,threshVal,inequal,weightedError))
                # 存储信息
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()  # 重新分配内存
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


"""
#测试
dataMat,classLabels = loadSimpleData()
D = mat(ones((5,1))/5)
bestStump,minError,bestClassEst = buildStump(dataMat,classLabels,D)
print(bestStump)
print(minError)
print(bestClassEst)
"""


# 基于单层决策树得到多个弱分类器
def adaBoostTrainDS(dataArr, classLabels, numIt):  # 数据集，分类标签，迭代次数
    weakClassArr = []  # 存储得到的单层决策树的信息
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # 一开始每个数据的概率相等
    aggClassEst = mat(zeros((m, 1)))  # 记录每个数据点的类别估计累计值

    for i in range(numIt):  # 每一次迭代
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #print("D: ", D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 更新每个分类器的权值，max(error,1e-16)确保在没有发生错误是出现除零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print("classEst: ", classEst.T)

        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # 更新D，这是exp的指数部分
        D = multiply(D, exp(expon))
        D = D / D.sum()

        aggClassEst += alpha * classEst  # 累加每一个预测值和该分类器权重的乘积
        #print("aggClassEst: ", aggClassEst.T)
        aggError = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))  # 将预测错的位置置1，正确的置0
        # 这一行可以用以下两行被注释的代码代替， # sign(t)函数：在t>=0是，sign(t)=1;在t<0是，sign(t)=-1 符号函数
        # aggError = mat(zeros((m,1)))
        # aggError[sign(aggClassEst) != mat(classLabels).T] = 1
        errorRate = aggError.sum() / m  # 错误比例
        #print("total error rate: ", errorRate, '\n')
        if errorRate == 0.0:
            break  # 如果错误比例已经为0则终止循环
    return weakClassArr,aggClassEst

"""
#测试
dataMat,classLabels = loadSimpleData()
weakClassArr,aggClassEst = adaBoostTrainDS(dataMat,classLabels,9)
print(weakClassArr)
print(weakClassArr[0]['thresh'])
"""

# 测试算法：基于AdaBoost的分类
# 分类函数
def adaClassify(datToClass,classifierArr): #这里和书上的代码有出入，更新如下
    #print(type(classifierArr))
    #print(len(classifierArr))
    #print(classifierArr[0])
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    lenClassfier = len(classifierArr[0])
    classifierDict = classifierArr[0] #这是最佳单层决策树的信息

    for i in range(lenClassfier):  # 对每一个分类进行遍历预测
        #print(i)
        #print(classifierDict[i])
        classEst = stumpClassify(dataMatrix,classifierDict[i]['dim'],classifierDict[i]['thresh'],classifierDict[i]['ineq'])
        aggClassEst += classifierDict[i]['alpha'] * classEst
        #print(aggClassEst)
    return sign(aggClassEst)

"""
# 测试
dataMat,classLabels = loadSimpleData()
weakClassArr = adaBoostTrainDS(dataMat,classLabels,30) #这里我就用一个参数接受了两个返回值，
# 所以导致adaClassify()函数和书上代码不一样但是本质上是一样的
print(weakClassArr)
classifiedLabels = adaClassify([[5,5],[0,0]],weakClassArr)
print(classifiedLabels)
"""

#曾经用Logistic回归预测疝病马是否存活，现在利用单层决策树和adaBoost预测
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = [];labelMat = []
    fr = open(filename)

    for line in fr.readlines():
        lineArr = []
        currLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(currLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))
    return dataMat,labelMat

"""
#测试
dataArr,labelArr = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch07\\horseColicTraining2.txt')
classifierArray = adaBoostTrainDS(dataArr,labelArr,30) #注意，以后如果函数返回的是两个参数，那就用两个参数接受，否则容易出错
testArr,testLabelArr = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch07\\horseColicTest2.txt')
m = shape(testArr)[0]
prediction = adaClassify(testArr,classifierArray)
errArr = mat(ones((m,1)))
sumerr = errArr[prediction != mat(testLabelArr).T].sum()
errRate = sumerr / m
print(1-errRate) #这是正确率
"""


#ROC曲线的绘制及AUC（曲线下的面积）计算函数，计算不同的阈值情况下，假阳率（X轴），真阳率（Y轴），的曲线
def plotROC(predStrengths,classLabels):#predStrengths是所有分类器对应预测值的和，也就是aggClassEst
    cur = (1.0,1.0) #浮点数二元组，初始值是（1.0,1.0），该元组保留的是绘制光标的位置
    ySum = 0.0 #用于计算AUC的值
    numPosClas = sum(array(classLabels)==1.0) #计算真实数据中标签为1的个数
    yStep = 1 / float(numPosClas) #x步长
    xStep = 1 / float(len(classLabels)-numPosClas) #y步长
    sortedIndicies = predStrengths.argsort() #从小到大排序

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    for index in sortedIndicies.tolist()[0]:#.tolist将numpy的元组或矩阵改成列表形式
        if classLabels[index] == 1.0:#数据集标签中为1时，改变Y轴坐标
            delX = 0;delY = yStep
        else:#数据集标签中为1时，改变X轴坐标
            delX = xStep;delY = 0
            ySum += cur[1] #计算每个数据y的高度，因为小矩形的宽度是xStep，AUC=xStep*ySum
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate');plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)

"""
#测试
dataArr,labelArr = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch07\\horseColicTraining2.txt')
classifierArray,aggClaaEst = adaBoostTrainDS(dataArr,labelArr,10)
plotROC(aggClaaEst.T,labelArr)
"""


"""
#一下是和书上一致的代码
from numpy import *
import matplotlib.pyplot as plt


##基于单层决策树构建弱分类器
def loadSimpleData():
    dataMat = matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


# 单层决策树生成函数
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # 数据集，第dimen+1个特征，阈值，两种阈值过滤模式
    retArray = ones((shape(dataMatrix)[0], 1))  # 预测出数据集标签向量
    if threshIneq == 'lt':  # 阈值过滤方式1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0  # 某一列的数值小于等于阈值，则赋值为-1.0（阈值一边赋值1，另一边赋值-1）
    else:  # 阈值过滤方式2
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):  # 数据集，数据集标签，基于数据的权重向量
    dataMatrix = mat(dataArr);
    labelsMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0;
    bestStump = {};
    bestClassEst = mat(zeros((m, 1)))
    # numSteps用于在特征的所有值上面遍历，bestStump存储给定权重D时所得的最佳单层决策树的相关信息，bestClassEst是预测的数据集标签向量
    minError = inf  # 正无穷，最小的误差设定为正无穷，然后重新赋值

    for i in range(n):  # 第一层循环：对数据集的每一个特征
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 步长

        for j in range(-1, int(numSteps) + 1):  # 第二层循环：每一个步长

            for inequal in ['lt', 'gt']:  # 第三层循环：对每一个不等号，这是两种不同的过滤模式
                threshVal = (rangeMin + float(j) * stepSize)  # 阈值计算
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 预测每一个数据的标签
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelsMat] = 0  # 将错误向量中正确分类的位置置0
                weightedError = D.T * errArr  # 这是一个数，算出错误的权重
                # print("split:dim %d,thresh %f,thresh inequal:%s,the weighted error is %f"%(i,threshVal,inequal,weightedError))
                # 存储信息
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()  # 重新分配内存
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


"""
#测试
dataMat,classLabels = loadSimpleData()
D = mat(ones((5,1))/5)
bestStump,minError,bestClassEst = buildStump(dataMat,classLabels,D)
print(bestStump)
print(minError)
print(bestClassEst)
"""


# 基于单层决策树得到多个弱分类器
def adaBoostTrainDS(dataArr, classLabels, numIt):  # 数据集，分类标签，迭代次数
    weakClassArr = []  # 存储得到的单层决策树的信息
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # 一开始每个数据的概率相等
    aggClassEst = mat(zeros((m, 1)))  # 记录每个数据点的类别估计累计值

    for i in range(numIt):  # 每一次迭代
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #print("D: ", D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 更新每个分类器的权值，max(error,1e-16)确保在没有发生错误是出现除零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print("classEst: ", classEst.T)

        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # 更新D，这是exp的指数部分
        D = multiply(D, exp(expon))
        D = D / D.sum()

        aggClassEst += alpha * classEst  # 累加每一个预测值和该分类器权重的乘积
        #print("aggClassEst: ", aggClassEst.T)
        aggError = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))  # 将预测错的位置置1，正确的置0
        # 这一行可以用以下两行被注释的代码代替， # sign(t)函数：在t>=0是，sign(t)=1;在t<0是，sign(t)=-1 符号函数
        # aggError = mat(zeros((m,1)))
        # aggError[sign(aggClassEst) != mat(classLabels).T] = 1
        errorRate = aggError.sum() / m  # 错误比例
        #print("total error rate: ", errorRate, '\n')
        if errorRate == 0.0:
            break  # 如果错误比例已经为0则终止循环
    return weakClassArr,aggClassEst


"""
#测试
dataMat,classLabels = loadSimpleData()
adaBoostTrainDS(dataMat,classLabels,9)
"""


# 测试算法：基于AdaBoost的分类
# 分类函数
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))

    for i in range(len(classifierArr)):  # 对每一个分类进行遍历预测
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        #print(aggClassEst)
    return sign(aggClassEst)

"""
# 测试
dataMat,classLabels = loadSimpleData()
weakClassArr ,aggClassEst = adaBoostTrainDS(dataMat,classLabels,30)
classifiedLabels = adaClassify([[0,0],[5,5]],weakClassArr)
print(classifiedLabels)
"""

#曾经用Logistic回归预测疝病马是否存活，现在利用单层决策树和adaBoost预测
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = [];labelMat = []
    fr = open(filename)

    for line in fr.readlines():
        lineArr = []
        currLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(currLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))
    return dataMat,labelMat

"""
#测试
dataArr,labelArr = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch07\\horseColicTraining2.txt')
classifierArray,aggClassEst = adaBoostTrainDS(dataArr,labelArr,30)
testArr,testLabelArr = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch07\\horseColicTest2.txt')
m = shape(testArr)[0]
prediction = adaClassify(testArr,classifierArray)
errArr = mat(ones((m,1)))
sumerr = errArr[prediction != mat(testLabelArr).T].sum()
errRate = sumerr / m
print(errRate)
"""


#ROC曲线的绘制及AUC（曲线下的面积）计算函数，计算不同的阈值情况下，假阳率（X轴），真阳率（Y轴），的曲线
def plotROC(predStrengths,classLabels):#predStrengths是所有分类器对应预测值的和，也就是aggClassEst
    cur = (1.0,1.0) #浮点数二元组，初始值是（1.0,1.0），该元组保留的是绘制光标的位置
    ySum = 0.0 #用于计算AUC的值
    numPosClas = sum(array(classLabels)==1.0) #计算真实数据中标签为1的个数
    yStep = 1 / float(numPosClas) #x步长
    xStep = 1 / float(len(classLabels)-numPosClas) #y步长
    sortedIndicies = predStrengths.argsort() #从小到大排序

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    for index in sortedIndicies.tolist()[0]:#.tolist将numpy的元组或矩阵改成列表形式
        if classLabels[index] == 1.0:#数据集标签中为1时，改变Y轴坐标
            delX = 0;delY = yStep
        else:#数据集标签中为1时，改变X轴坐标
            delX = xStep;delY = 0
            ySum += cur[1] #计算每个数据y的高度，因为小矩形的宽度是xStep，AUC=xStep*ySum
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate');plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)

"""
#测试
dataArr,labelArr = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch07\\horseColicTraining2.txt')
classifierArray,aggClaaEst = adaBoostTrainDS(dataArr,labelArr,10)
plotROC(aggClaaEst.T,labelArr)
"""
"""


