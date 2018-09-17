####3.1###
#计算给定数据集的香农熵，为了计算信息增益，取最大信息增益的特征值作为数据分类的依据
from cmath import log

#计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #数据集的个数
    labelCounts = {} #存储每一类标签和出现的次数，为后面计算香农熵
    for fentVec in dataSet:
        currentLabel = fentVec[-1]  #最后一列作为标签
        #labelCounts[currentLabel] = labelCounts.get(currentLabel,0) + 1 #这行代码可以由以下三行代替
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0 #初始化香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]/numEntries) #每一类标签出现的概率
        shannonEnt -= prob * (log(prob,2)).real #计算信息熵 ，log函数返回的是实数+虚数的格式，2表示以2为底
    return shannonEnt #熵越高混合的数据就越多

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers'] #表示每个数据集前两个特征的含义
    return dataSet,labels

"""
#测试
myDat,labels = createDataSet()
shannonEnt = calcShannonEnt(myDat)
print(shannonEnt)
"""

#按照指定的特征指定的值划分数据集
def splitDateSet(dataSet,axis,value):
#dataSet是数据集，axis是表示按照第几列也就是第几个(axis+1)特征划分,value是返回特征中包含某个特定值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value: #如果第axis列的值等于value
            #以下两行代码就是去除了第(axis+1)个特征中包含value的值，在是value的情况下留下其他特征的所有数据，也就是降了一维
            reducedFeatVec = featVec[:axis] #reducedFeatVec包含该特征之前的元素
            reducedFeatVec.extend(featVec[axis+1:]) #reducedFeatVec包含该特征之后的元素
            retDataSet.append(reducedFeatVec)
    return retDataSet

"""
#测试
myDat,labels = createDataSet()
print(len(myDat[0]))
retDataSet = splitDateSet(myDat,1,0)
print(retDataSet)
"""

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 #最后一个是分类的标签，所以特征就列数-1
    baseEntropy = calcShannonEnt(dataSet) #计算划分前信息熵作为最基本的信息熵
    baseInfoGain = 0.0 #初始化信息增益
    baseFeature = -1 #初始化baseFeature
    for i in range(numFeatures): #i表示第i+1个特征值
        featureList = [example[i] for example in dataSet] #将dataset中每一个元素，添加到列表当中
        uniqueVals = set(featureList) #找出第i列不同元素集合
        newEntropy = 0.0
        #计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDateSet(dataSet,i,value) #按照指定的特征指定的值划分数据集
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        #计算增益
        infoGain = baseEntropy - newEntropy
        #选择最大增益和确定最大增益的属性
        if infoGain > baseInfoGain:
            baseInfoGain = infoGain
            baseFeature = i
    return baseFeature

"""
#测试
myDat,labels = createDataSet()
baseFeature = chooseBestFeatureToSplit(myDat)
print(baseFeature)
"""

#某属性中出现次数最多的类别
import operator

def majorityCnt(classList):
    classCount = {}
    for vote in classList: #如果vote在标签列表中
        classCount[vote] = classCount.get(vote,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #逆序排序
    return sortedClassCount[0][0]

#创建树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] #标签列转成列表形式
    #递归函数停止的第一个条件是：所有的类标签完全相同
    if classList.count(classList[0]) == len(classList): #标签列表中第一个元素的个数等于列表的长度，表示只有一种标签
        return classList[0]
    #递归函数停止的第二个条件：使用完了所有的特征，仍然不能将数据集划分成只包含唯一类别的分组
    #由于第二个条件无法简单返回唯一的类标签，所以使用前面majorityCnt()选出最多的类别作为返回值
    #print(dataSet[0])
    if len(dataSet[0]) == 1: #dataSet[0]矩阵中第一条数据，如果矩阵的第一条数据的个数为1，也就是只剩最后一列标签类，也就是特征用完了
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #调用函数算出分类的最好特征所在的列
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    #del labels[bestFeat] #书上代码有误，更改如下
    subLabels = labels[:] #先定义一个子标签以防止更改了原来的labels
    del subLabels[bestFeat] #删除这个最好的标签
    featValues = [example[bestFeat] for example in dataSet] #取最好的特征列的所有特征值
    uniqueVals = set(featValues) #最好特征列的所有特征值的集合，不重复的元素
    for value in uniqueVals:
        #subLabels = labels[:]   #书上在此定义，但是应该在前面定义，防止改变了原来的labels
        myTree[bestFeatLabel][value] = createTree(splitDateSet(dataSet,bestFeat,value),subLabels) #递归创建树
    return myTree

"""
#测试
myDat,labels = createDataSet()
myTree = createTree(myDat,labels)
print(myTree)
"""

#使用决策树的分类函数
def classify(inputTree,featLabels,testVec): #树，所有特征，要分类的数据
    firstStr = list(inputTree.keys())[0]  #每一次调用都获得字典的键，也就是每一次获得树的一个非叶节点
    #print(firstStr)
    #print(type(featLabels[0]))
    secondDict = inputTree[firstStr] #字典中键对应的值，也是字典
    featIndex = featLabels.index(firstStr) #因为不知实际的属性存储的位置，所以索引出下标
    for key in secondDict.keys(): #在字典所有的键中
        if testVec[featIndex] == key: #如果要测试的分类数据的某个特征的特征值是等于key
            if type(secondDict[key]).__name__ == 'dict': #如果是字典
                classLabel = classify(secondDict[key],featLabels,testVec) #则递归调用分类函数
            else:
                classLabel = secondDict[key] #否则哟啊返回的标签就是字典的值
    return classLabel

"""
#测试
myDat,labels = createDataSet()
#print(labels)
myTree = createTree(myDat,labels)
print(myTree)
classLabel = classify(myTree,labels,[1,1])
#print(classLabel)
"""

#构造决策树很费时间，数据集很大就更费时间，用创建好的决策树解决分类问题
#使用pickle模块存储决策树，pickle模块序列化对象
def storeTree(inputTree,filename):
    import pickle
    #fw = open(filename,'w') #'w'改成'wb'不然会报错TypeError: write() argument must be str, not bytes
    fw = open(filename, 'wb')
    pickle.dump(inputTree,fw) #将inputTree序列化到fw文件中
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

"""
#myDat,labels = createDataSet()
#myTree = createTree(myDat,labels)
#storeTree(myTree,r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch03\\classifierStorage.txt')
#grabTree = grabTree(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch03\\classifierStorage.txt')
#print(grabTree)
"""

#使用决策树预测隐形眼镜的类型
def predictLenses(filename):
    fr = open(filename)
    lenses = [inst.strip().split('\t') for inst in fr.readlines()] #按照tab键划分
    print(lenses)
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    lensesTree = createTree(lenses,lensesLabels)
    return lensesTree

"""
#测试
lensesTree = predictLenses(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch03\\lenses.txt')
print(lensesTree)
"""

#可以将lensesTree的信息添加到teePlotter.py的retrieveTree（）函数中的listOfTree列表中，然后改变参数i可以绘制决策树图







