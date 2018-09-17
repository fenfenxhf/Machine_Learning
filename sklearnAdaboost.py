from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score

#数据集处理
def loadData(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filename)

    for line in fr.readlines():
        lineArr = []
        currLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(currLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))
    return dataMat, labelMat

x_train,y_train = loadData(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch07\\horseColicTraining2.txt')
x_test,y_test = loadData(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch07\\horseColicTest2.txt')

#print(x_train)
#print(y_train)
#print(x_test)
#print(y_test)

clf = AdaBoostClassifier()

clf.fit(x_train,y_train)

y_predict = clf.predict(x_test)

scores = clf.score(x_test,y_test)

accuracyScore = accuracy_score(y_test,y_predict)

print(scores)
print(accuracyScore)

#算法精度没有adaboost.py的算法精度高