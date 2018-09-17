from numpy import *
import random
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score

"""
#以下是各种回归
from sklearn.tree import DecisionTreeRegressor #决策树回归
from sklearn.linear_model import Ridge #岭回归
from sklearn.svm import SVR #支持向量机回归
from sklearn.neighbors import KNeighborsRegressor #KNN回归
from sklearn.ensemble import RandomForestRegressor #随机森林回归
from sklearn.ensemble import AdaBoostRegressor #Adaboost回归
from sklearn.ensemble import GradientBoostingRegressor #GBRT回归
from sklearn.ensemble import BaggingRegressor #Bagging回归
from sklearn.tree import ExtraTreeRegressor #极端随机树回归
"""

#数据准备
def loadData(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1  # 特征数
    dataMat = []
    labelMat = []
    dataIntegrity = [] #存储完整的数据
    fr = open(filename)

    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat+1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr[:-1])
        labelMat.append(float(curLine[-1]))
        dataIntegrity.append(lineArr)
    return dataMat, labelMat,dataIntegrity

xArr,yArr ,dataIntegrity = loadData(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch08\\abalone.txt')
#print(len(yArr)) #4177条数据
#print(xArr[0])
#print(dataIntegrity[0])
"""
#随机选800条数据做测试集
trainSet = list(range(len(yArr))) #[0,4177)
testSet = [] #保存测试集的索引
x_train = []; y_train = []
x_test = []; y_test = []
for i in range(800):
    randIndex = int(random.uniform(0,len(trainSet)))
    testSet.append(trainSet[randIndex])
    del trainSet[randIndex]

#训练集
for dataIndex in trainSet:
    x_train.append(xArr[dataIndex])
    y_train.append(yArr[dataIndex])

#测试集
for dataIndex in testSet:
    x_test.append(xArr[dataIndex])
    y_test.append(yArr[dataIndex])

print(x_train)
print(y_train)
print(x_test)
print(y_test)
"""

clf = KernelRidge()

clf.fit(xArr,yArr)
y_predict = clf.predict(xArr)
#y_predict_int = []
#for i in range(len(y_predict)):
#    y_predict_int.append(int(y_predict[i]))

print(yArr)
print(y_predict)
print(clf.score(xArr,yArr))



