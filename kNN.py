######2.1#####
from numpy import *  #numpy科学计算包
import operator  #operator运算符模块

#创建数据集以及每个数据对应标签
def creatDataSet():
    global group
    global labels
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#实现k-近邻算法
def classify0(inx,dataSet,labels,k): #inx为要分类的坐标如[0,0.2]，dataSet为array，labels为标签，k为前k个最相近的
    #计算距离
    dataSetSize = dataSet.shape[0] #0表示行数，1表示列数
    #diffMat = tile(inx,(dataSetSize,1))-dataSet #tile将inx沿着y方向复制dataSetSize次，沿x方向复制1次（相当于不变）
    diffMat = dataSet - inx #和上面这句话效果一致
    sqDiffMat = diffMat ** 2 #数组**2表示对应元素的平方
    sqDistances = sqDiffMat.sum(axis=1) #使用0值表示沿着每一列或行标签\索引值向下执行方法,使用1值表示沿着每一行或者列标签横向执行对应的方法
    distances = sqDistances ** 0.5 #数组**0.5表示对应元素的平方根

    sortedDistIndicies = distances.argsort() #将distances从小到大排列并返回索引的下标，并非返回元素本身
    classCount = {}
    #确定前k个点所在类别的出现频率
    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]] #获得字典classCount的key，从labels中获得
        classCount[voteILabel] = classCount.get(voteILabel,0) + 1 #给key赋值，classCount.get(voteILabel,0)
        # 就是如果字典中有voteILabel，则返回对应的value,没有则返回0

    #将前k个点所在类别的出现频率按照从大到小排序
    #sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    sortedClassCount = sorted(classCount.items(),key=lambda x:x[1],reverse=True) #和上面那句话效果一致
    return sortedClassCount[0][0]

"""
#测试
group,labels = creatDataSet()
kNN = classify0([1,1.2],group,labels,3)
print(kNN)
"""


####2.2####
#将文本记录转化为Numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3)) #创建numberOfLines*3的零矩阵(数组类型)
    classLabelVector = [] #创建标签向量列表
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3] #将第Index+1行的前三个特征添加到returnMat
        classLabelVector.append(int(listFromLine[-1])) #将字符串转化成整形表示
        index += 1
    return returnMat,classLabelVector


"""自练的代码
def file3matrix(filename):
    fr = open(filename)
    #m = len(fr.readlines())
    dataList = []
    labelMat = []
    for line in fr.readlines():
        #print("******")
        currLine = line.strip().split('\t')
        #print(currLine)
        #currDat = currLine[0:3]
        dataList.append([float(currLine[0]),float(currLine[1]),float(currLine[2])])
        labelMat.append(int(currLine[-1]))
    dataMat = mat(dataList)
    return dataMat,labelMat
"""

""""
#测试
datingDataMat,datingLabels = file2matrix(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\'
                                         r'machinelearninginaction\\Ch02\\datingTestSet2.txt')
print(datingDataMat)
#print(mat(datingDataMat))
print(datingLabels)
"""

"""
#####使用matplotlib创建散点图
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],c = 'r')
#plt.show()
"""

####归一化特征值####
def autoNorm(dataSet):
    minVals = dataSet.min(0) #0表示沿着列向下执行,求这一列的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    #print(normDataSet)
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1)) #x轴1倍，y轴m倍
    #normDataSet = dataSet - minVals #与上句等效
    normDataSet = normDataSet / tile(ranges,(m,1)) #newValue = (oldValue-min)/(max-min)
    #normDataSet = normDataSet / ranges #与上句等效
    return normDataSet,ranges,minVals

"""
#测试
datingDataMat,datingLabels = file2matrix(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\'
                                         r'machinelearninginaction\\Ch02\\datingTestSet2.txt')
normDataSet,ranges,minVals = autoNorm(datingDataMat)
print(normDataSet)
print(ranges)
print(minVals)
"""

####分类器针对约会网站的测试代码####
def datingClassTest():
    hoRatio = 0.1 #选取数据的10%作为测试数据
    datingDataMat, datingLabels = file2matrix(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\'
                                              r'machinelearninginaction\\Ch02\\datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio) #测试数据数量
    errorCount = 0.0
    for i in range(numTestVecs): #取前面10%的数据作为测试集
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with:%d,the real answer is:%d"%(classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    totalErrorRate = errorCount / float(numTestVecs)
    print(totalErrorRate)

#datingClassTest()

###构建完整的可用预测系统
def classfyPerson():
    resultList = ['not at all','in small doses','in large doses']
    ffMiles = float(input("frequent flier miles earned per year?"))
    percentTats = float(input("percentage of time spent playing video games?"))
    iceCream = float(input("liters of ice cream earned per year?"))
    datingDataMat, datingLabels = file2matrix(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\'
                                              r'machinelearninginaction\\Ch02\\datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classfierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print(classfierResult)
    print("You will probably like this person:%s"%resultList[classfierResult - 1])
    #datingTestSet2.txt文件中标签是1,2,3，而索引是从0开始

#classfyPerson()

####2.3####
#将图像（32*32）转化为1*1024的向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32): #遍历32行
        lineStr = fr.readline()
        for j in range(32): #遍历32列
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#returnVect = img2vector(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch02\\digits\\testDigits\\0_0.txt')
#print(returnVect[0,0:31])

#手写数字识别系统的测试代码
from os import listdir #列出文件

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch02\\digits\\trainingDigits')
    m = len(trainingFileList) #文件个数
    trainMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainMat[i,:] = img2vector(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\'
                                   r'machinelearninginaction\\Ch02\\digits\\trainingDigits\\%s'%fileNameStr)

    testFileList = listdir(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\'
                           r'machinelearninginaction\\Ch02\\digits\\testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\'
                                     r'machinelearninginaction\\Ch02\\digits\\testDigits\\%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainMat,hwLabels,3)
        #print('the classifier came back with:%d,the real answer is:%d'%(classifierResult,classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1
            print('the classifier came back with:%d,the real answer is:%d' % (classifierResult, classNumStr))
    print('total error rate is:%f'%(errorCount/float(mTest)))

handwritingClassTest()












