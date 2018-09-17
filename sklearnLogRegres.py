from sklearn.linear_model import LogisticRegression
from sklearn import metrics #sklearn中的评估模块，主要用于对结果好坏的评测
from sklearn.metrics import accuracy_score

def loadData(filename):
    fr = open(filename).readlines()
    data = [];target = []
    for line in fr:
        currLine = line.strip().split('\t')
        data.append([float(inst) for inst in currLine[:-1]])
        target.append(float(currLine[-1]))
    return data,target

train_data,train_target = loadData(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\'
                                   r'Ch05\\horseColicTraining.txt')
test_data,test_target = loadData(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\'
                                   r'Ch05\\horseColicTest.txt')
"""
#测试数据集是否正确
print(train_data)
print(train_target)
print(test_data)
print(test_target)
"""

clf = LogisticRegression()
clf.fit(train_data,train_target)

predicted = clf.predict(test_data)
expected = test_target

mcr = metrics.classification_report(expected,predicted)
mcm = metrics.confusion_matrix(expected,predicted)
accuracyScore = accuracy_score(test_target,predicted)

print(mcr) #打印信息如下：
#                   precision    recall  f1-score   support

#       0.0 (标签)      0.54      0.65      0.59        20
#       1.0 (标签)      0.84      0.77      0.80        47

#      avg / total      0.75      0.73      0.74        67
#精度（precision）= 正确预测的个数(TP)/被预测正确的个数(TP+FP)
#召回率（recall）= 正确预测的个数(TP)/预测的个数(TP+FN)
#F1 = 2*精度*召回率/(精度+召回率)

print(mcm) #打印信息如下：
#[[13  7] #真值为0，预测为0的有13个；真值为0，预测值为1的有7个
#[11 36]] #真值为1，预测值为0的有11个；真值为1，预测值为1的有36个

print(accuracyScore) #打印：0.7313432835820896
#sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
#normalize：默认值为True，返回正确分类的比例；如果为False，返回正确分类的样本数






