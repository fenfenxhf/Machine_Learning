
"""
#测试：鸢尾花数据集
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#load data
iris=load_iris()
print(iris)
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
mnb=GaussianNB()
mnb.fit(X_train,y_train)
print(mnb.predict(X_test))
print(y_test)
print(mnb.score(X_test,y_test))

"""

#使用sklearn来对垃圾邮件过滤
from numpy import *
import re  # 正则表达式
from sklearn.naive_bayes import GaussianNB

def textParse(bigString):
    #需要导入正则表达式包
    #listOfTokens = re.split('[\.| ]',bigString) #\w表示字母数字，*表示0个1个或多个，？表示0或1个，+表示1或多个
    #上面这行代码可用下面两行代码代替，效果一样
    regEx = re.compile('[\.| ]') #编译正则表达式中'.'或者空格
    listOfTokens = regEx.split(bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] #实际中一般会过滤掉长度小于3的字符串

def bagOfWords2Vec(vocabList,inputSet): #词袋模型，vocabList是createVocabList（）中的vocabSet，inputSet是文档
    returnVec = [0] * len(vocabList)  #returnVec是词向量，定义一个和vocabList一样长的全0列表，1*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 #在vocabList索引该词汇的位置，并将该位置的returnVec计数
    return returnVec

def createVocabList(dataSet):
    vocabSet = set([])  #定义一个空集合
    #print(type(vocabSet))  #list
    for document in dataSet:
        vocabSet = vocabSet | set(document) #将每一个文档不重复词条添加到空集合vocabSet中
    return list(vocabSet)

#数据集分析
def dataSetTran():
    import random
    docList = [];classList = [];fullText = []
    #导入并解析文件
    for i in range(1,26): #spam和ham文件夹中各有25个文件，从1开始是为了方便后面使用i
        wordList = textParse(open(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction'
                                  r'\\Ch04\\email\\spam\\%d.txt'%i).read())
        docList.append(wordList) #[[],[],[],[]....]
        fullText.extend(wordList) #[.....]
        classList.append(1) #垃圾文件标1
        wordList = textParse(open(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction'
                                  r'\\Ch04\\email\\ham\\%d.txt'%i).read())
        docList.append(wordList)  # [[],[],[],[]....]
        fullText.extend(wordList)  # [.....]
        classList.append(0)  # 垃圾文件标0 ，一次循环后是1 0 1 0 ...这样相间
    vocabList = createVocabList(docList) #创建词集
    #随机构建训练集
    trainingSet = list(range(50));testSet = [] #书上代码有误，这里要把trainingSet转化成List形式
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        #uniform() 方法将随机生成下一个实数，它在 [0,len(trainingSet) 范围内
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = [];trainClasses = [];testMat = [];testClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    for docIndex in testSet:
        testMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
        testClasses.append(classList[docIndex])
    return trainMat,trainClasses,testMat,testClasses

#测试
trainMat,trainClasses,testMat,testClasses = dataSetTran()
#print(trainMat)
#print(trainClasses)
#print(len(testMat))
#print(testClasses)

x_train = array(trainMat)
y_train = array(trainClasses)
x_test = array(testMat)
y_test = array(testClasses)

clf = GaussianNB()
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
print(y_predict)
y_predict_pro = clf.predict_proba(x_test)
print(y_predict_pro)
score = clf.score(x_test,y_test)
print(score)

