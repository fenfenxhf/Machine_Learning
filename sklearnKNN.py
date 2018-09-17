from numpy import *
import numpy as np
from os import listdir
from sklearn import neighbors

#将图像（32*32）转化为1*1024的向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32): #遍历32行
        lineStr = fr.readline()
        for j in range(32): #遍历32列
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#
def HWdataSet(folder):
    hwLabels = []
    FileList = listdir(folder)
    m = len(FileList)  # 文件个数
    dataMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = FileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        dataMat[i, :] = img2vector(folder + '\\' + fileNameStr) #字符串的连接+
    return dataMat,hwLabels

#测试
dataMat_train , hwLabels_train = HWdataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\'
                                           r'machinelearninginaction\\Ch02\\digits\\trainingDigits')
knn = neighbors.KNeighborsClassifier(n_neighbors=3) #实例对象
knn.fit(dataMat_train,hwLabels_train) #训练

dataMat_test , hwLabels_test = HWdataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\'
                                         r'machinelearninginaction\\Ch02\\digits\\testDigits')

pre = knn.predict(dataMat_test) #预测测试集的标签
#error_num = np.sum(pre != hwLabels_test) #计算错误的个数
#num = len(dataMat_test)
#error_rate = error_num / num
#print(1 - error_rate)

#以下两行等效上面注释的四行
score = knn.score(dataMat_test,hwLabels_test,sample_weight=None) #计算正确的概率
print(score)

probility = knn.predict_proba(dataMat_test) #给出测试集每个结果的可能性（概率），返回矩阵形式
print(probility)

#比较了和KNN.py的算法精确度，本次是KNN.py较高

