"""
#下面是以iris鸢尾花数据集为例讲述SVM算法
from sklearn import svm,datasets
import numpy as np

clf = svm.SVC() #调用SVC()

iris = datasets.load_iris() #载入鸢尾花数据
X = iris.data
y = iris.target

#clf.fit(X,y)#训练
clf.fit(X,y)

#pre_y = clf.predict(X[5:10]) #为了方便我们只预测几个数据[5,9)
pre_y = clf.predict(X[5:10])
print(pre_y)
print(y[5:10])

test = np.array([[5.1,2.9,1.8,3.6]])

test_y = clf.predict(test)

print(test_y)
"""

#手写问题的回顾（scikit-learn）
#数据的处理
from numpy import *
from os import listdir
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score

def image2Vec(filename):
    returnVec = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        currLine = fr.readline()
        for j in range(32):
            returnVec[0,32*i+j] = int(currLine[j])
    return returnVec

def loadData(dirname):
    fileList = listdir(dirname)
    m = len(fileList)
    dataMat = zeros((m,1024))
    labelMat = []

    for i in range(m):
        fileNameStr = fileList[i]
        fileName = fileNameStr.split('.')[0]
        label = int(fileName.split('_')[0])
        labelMat.append(label)
        dataMat[i,:] = image2Vec(r'%s\\%s'%(dirname,fileNameStr))
    return dataMat,labelMat

x_train,y_train = loadData(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch02\\digits\\trainingDigits')
x_test,y_test = loadData(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch02\\digits\\testDigits')

#print(x_train)
#print(y_train)
#print(x_test)
#print(y_test)

#使用scikit-learn中的svm处理手写数字识别问题
clf = svm.SVC()

clf.fit(x_train,y_train)

y_predict = clf.predict(x_test)
print(y_test)
print(y_predict)

score_train = clf.score(x_train,y_train)
print(score_train)
score_test = clf.score(x_test,y_test)
print(score_test)

accuracyscore = accuracy_score(y_test,y_predict)
print(accuracyscore) #和上面score_test结果一样

print(clf.support_) #返回支持向量的索引
print(clf.support_vectors_) #返回支持向量
print(clf.n_support_) #返回每一个标签支持向量的个数






