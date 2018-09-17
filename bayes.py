#朴素贝叶斯用于文档分类的常用算法
#朴素贝叶斯中假设：特征独立，特征同等重要

from numpy import *
import re  # 正则表达式
#from cmath import log
import feedparser

#词表到向量的转化
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1表示侮辱性言论，0表示非侮辱性言论
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])  #定义一个空集合
    #print(type(vocabSet))  #list
    for document in dataSet:
        vocabSet = vocabSet | set(document) #将每一个文档不重复词条添加到空集合vocabSet中
    return list(vocabSet)

"""
#测试
postingList,classVec = loadDataSet() #注意如果函数是返回两个值，那调用函数时候就要用两个值来接受，否则就会报错
myDataSet = createVocabList(postingList)
print(myDataSet)
"""

def setOfWords2Vec(vocabList,inputSet): #词集模型，vocabList是createVocabList（）中的vocabSet，inputSet是文档
    returnVec = [0] * len(vocabList)  #returnVec是词向量，定义一个和vocabList一样长的全0列表,对应着上面list(vocabSet)
    #print(returnVec) #打印出来是列表形式[0, 0, 0, 0,...]
    #以下两个打印出来都是一维数组
    #returnVec1 = zeros((1,len(vocabList)))
    #print(returnVec1) #打印出来是1*len(vocabList)的向量,[[0. 0. 0. 0. ...]]
    #returnVec2 = zeros(len(vocabList))
    #print(returnVec2) #打印出来是[0. 0. 0. 0. 0. ...]
    for word in inputSet:#inputSet是给定的一个文档
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 #在vocabList索引（索引是从0开始的）该词汇的位置，并将该位置的returnVec置1
        else:
            print('the word:%s is not in my Vocabulary!'%word)
    return returnVec

def bagOfWords2Vec(vocabList,inputSet): #词袋模型，vocabList是createVocabList（）中的vocabSet，inputSet是文档
    returnVec = [0] * len(vocabList)  #returnVec是词向量，定义一个和vocabList一样长的全0列表，1*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 #在vocabList索引该词汇的位置，并将该位置的returnVec计数
    return returnVec

"""
#测试
listOPosts,listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
print(myVocabList)
returnVec = setOfWords2Vec(myVocabList,listOPosts[0])
print(returnVec)
returnVecBag = bagOfWords2Vec(myVocabList,listOPosts[0])
print(returnVecBag)
"""

#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory): #trainMatrix是[[],[],[],[],[],[]]这样的6*len(vocabList)矩阵,6是文档数，
    #在setOfWords2Vec()函数中，我们将每篇文档转化成了词向量，所有词向量的长度相等，也就是len(vocabList)
                                         # trainCategory是loadDataSet()中的classVec
    numTrainDocs = len(trainMatrix) #文档数
    numWords = len(trainMatrix[0])   #词表中词的个数
    pAbusive = sum(trainCategory) / float(numTrainDocs) #trainCategory中1的个数所占比例，也就是侮辱性文档所占比例P(1)
                                         #这是二分类问题,P(0) = 1 - P(1)。如果是多分类问题则需要修改代码
    #sum(trainCategory)其中trainCategory是列表形式，表明列表也可以求和
    p0Num = ones(numWords);p1Num = ones(numWords) #对文档分类时，要计算多个概率的乘积以获得文档属于某个类别的概率，
                                                  # 将所有的词出现次数初始化为1，为了避免其中有概率为0则乘积就为0
    p0Denom = 2.0;p1Denom = 2.0  #将分母初始化为2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1: #如果是侮辱性文档，也就是类别1
            p1Num += trainMatrix[i] #每一列对应元素相加，得到在侮辱性条件下，每个词出现的次数
            p1Denom += sum(trainMatrix[i]) #得到侮辱性文档的所有单词个数的总和
            # #每一行相加（相当于列表相加求和），得到每一篇文档的单词个数，如第一篇是7个
        else: #如果是非侮辱性文档，也就是类别0
            p0Num += trainMatrix[i] #每一列对应元素相加，在非侮辱性条件下，每个词出现的次数
            p0Denom += sum(trainMatrix[i]) #得到非侮辱性文档所有单词个数的总和
    p1Vect = log(p1Num/p1Denom) #这里的Log不能从cmath中导入，直接使用Numpy中的，不然会报错，用log是为了避免相乘时数太小而四舍五入为0了，避免溢出
    p0Vect = log(p0Num/p0Denom) #（每个单词出现在类别0的次数）/（类别0出现的所有句子的单词总和）”,
                                # 得到的就是p(w1|c0),p(w2|c0),...,p(wn|c0)
    return p0Vect,p1Vect,pAbusive

"""
#测试
listOPosts,listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts) #生成词集
trainMat = []
for postinDoc in listOPosts: #取每一篇文档
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc)) #将每一篇文档生成词向量
p0V,p1V,pAb = trainNB0(trainMat,listClasses)
print(p0V)
print(p1V)
print(pAb)
"""

#朴素贝叶斯分类函数，极大似然法，谁大判谁
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1): #vec2Classify要分类的向量，后面三个参数是trainNB0()中返回的三个概率
    #待分类的向量vec2Classify，矩阵向量对应相乘piVec,与其相似的那个值必定大
    #log(p(w|ci)*p(ci))=log(p(w0|ci)*p(w1|ci)*...p(wn|ci)*p(ci) = sum(log(p(wi|ci))) + log(p(ci))
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    #print(vec2Classify * p1Vec) #打印出来是len(vec2Classify)的矩阵
    if p1 > p0:
        return 1
    else:
        return 0

 #封装测试词集模型
def testingNBCiJi():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    print(p0V)
    print(p1V)
    testEntry   = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry)) #转化成数组形式，调用时要向量相乘
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    """
    testEntry = ["stupid","garbage"]
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,"classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))
    """

#封装测试词袋模型
def testingNBCiDai():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bagOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry   = ['love','my','dalmation']
    thisDoc = array(bagOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ["stupid","garbage"]
    thisDoc = array(bagOfWords2Vec(myVocabList, testEntry))
    print(testEntry,"classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))

#testingNBCiDai()
#testingNBCiJi()

#将很长的字符串转化成小写字母的单词列表
def textParse(bigString):
    #需要导入正则表达式包
    #listOfTokens = re.split('[\.| ]',bigString) #\w表示字母数字，*表示0个1个或多个，？表示0或1个，+表示1或多个
    #上面这行代码可用下面两行代码代替，效果一样
    regEx = re.compile('[\.| ]') #编译正则表达式中'.'或者空格
    listOfTokens = regEx.split(bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] #实际中一般会过滤掉长度小于3的字符串

"""
#测试
mysent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
mysentSplited = textParse(mysent)
print(mysentSplited)
"""

#测试算法：使用朴素贝叶斯进行交叉验证
#文件解析和完整的垃圾邮件测试函数
def spamTest():
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
    trainMat = [];trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSam = trainNB0(array(trainMat),array(trainClasses)) #训练得到
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSam) != classList[docIndex]:#用训练得到的参数放到分类函数中，得到测试集的分类是否正确
            errorCount += 1
    print("the error rate is: ",float(errorCount/len(testSet)))

#spamTest()
#spamTest()#因为是随机分类，所以两次结果不一定一样

#使用朴素贝叶斯分类器从个人广告中获得区域倾向
#计算词出现的概率并获得前30个高概率
def calcMostFreq(vocabList,fullText): #vocabList是词集（集合形式），fullText是所有的单词（可重复出现）
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token) #添加字典键值对，count()计数
    sortedFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True) #逆序排序
    return sortedFreq[:30] #前30个

#测试：怎么使用RSS源
"""
import feedparser
ny = feedparser.parse('http://view.news.qq.com/index2010/zhuanti/ztrss.xml') #腾讯新闻的RSS源
print(len(ny['entries'])) #打印50，访问所有的条目
"""

#RSS源分类器及高频词去除函数
def localWords(feed1,feed0):
    #import feedparser
    import random
    docList = [];classList = [];fullText = []
    #获取feed1和feed0最小长度
    minLen = min(len(feed1["entries"]),len(feed0["entries"]))
    #print(minLen)
    for i in range(minLen):
        wordList = textParse(feed1["entries"][i]['summary']) #每一次访问一条RSS源
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0["entries"][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)

    #去掉最高频的30个词
    top30Words = calcMostFreq(vocabList,fullText)
    #print(top30Words) #打印结果[('the', 104), ('and', 21),...],这是键值对形式
    for pairW in top30Words:
        if pairW[0] in vocabList: #pairW[0]是键，也就是词本身
            vocabList.remove(pairW[0])

    trainingSet = list(range(2*minLen));testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = [];trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSam = trainNB0(array(trainMat),array(trainClasses)) #训练得到
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSam) != classList[docIndex]:#用训练得到的参数放到分类函数中，得到测试集的分类是否正确
            errorCount += 1
    print("the error rate is: ",float(errorCount/len(testSet)))
    return vocabList,p0V,p1V

"""
#测试
ny=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss') #书上的RSS源无数据，这里随便找了几个RSS源代替
sf=feedparser.parse('http://view.news.qq.com/index2010/zhuanti/ztrss.xml')
vocabList,pSF,pNF = localWords(ny,sf)
"""

#分析数据，显示最具有表征性的词汇
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V = localWords(ny,sf)
    topNY =[];topSF =[]
    for i in range(len(p0V)):
        if p0V[i] > -0.6:
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -0.6:
            topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print("SF**"*20)
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**" * 20)
    for item in sortedNY:
        print(item[0])

"""
#测试
ny=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss') #该网址无数据
sf=feedparser.parse('http://view.news.qq.com/index2010/zhuanti/ztrss.xml')
getTopWords(ny,sf)
"""






