#数据集
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

#在dataSet中创建一个只有一个元素的项集
def createC1(dataSet):
    C1 = []
    for transaction in dataSet: #在数据集中的每一条记录
        for item in transaction: #每一条记录中的每一个元素
            if not [item] in C1: #如果这个元素不在C1中，则添加该元素
                C1.append([item])
    #print(C1) #打印[[1], [3], [4], [2], [5]]
    C1.sort() #从小到大排序
    return list(map(frozenset,C1)) #frozenset是指被冰冻的集合，不可改变。
    # set也是不可改变，但是frozenset可以作为字典的键值使用，而set不行

def scanD(D,Ck,minSupport):#数据集，候选项集列表Ck，最小支持度
    ssCnt = {} #存储候选项集每一项在数据集中出现的次数
    for tid in D: #遍历数据集
        for can in Ck: #遍历候选集
            if can.issubset(tid): #can是不是tid的子集
                if not can in ssCnt:#如果ssCnt中没有这个候选集
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1 #如果ssCnt中有这个候选集,则出现次数就加一
    numItems = float(len(D)) #数据集的个数
    retList = [] #存储满足最小支持度的候选集
    supportData = {} #存储所有候选集项和对应的支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0,key) #0是索引，1是列表的对象,也可以用以下一行代码实现
            #retList.append(key)
        supportData[key] = support
    return retList,supportData #返回单个物品的频繁项集列表和所有单个物品的支持度

"""
#测试
dataSet = loadDataSet()
print(dataSet)
C1 = createC1(dataSet)
print(C1)
D = list(map(set,dataSet))
L1,support0 = scanD(D,C1,0.5)
print(L1)
print(support0)
"""

#组织完整的Apriori算法:在大数据集上可能比较慢，使用Apriori算法找到频繁项集
#创建Ck
def aprioriGen(Lk,k): #包含k个元素的频繁项集列表；第k频项集中的k
    retList = []
    lenLk = len(Lk) #频繁项集的个数
    for i in range(lenLk):#遍历每一个频繁项集
        for j in range(i+1,lenLk):#两两组合遍历
            L1 = list(Lk[i])[:k-2];L2 = list(Lk[j])[:k-2]
            L1.sort();L2.sort()
            if L1 == L2:
                retList.append(Lk[i]| Lk[j]) #如果前k-2个项相同时，将两个元素合并，为了创建添加一个元素后得项集
    return retList
#eg， {23}->{01}和{12}->{03} 合并为{2}->{013};因为右部前k-2相同，为了避免重复的合并操作，同上面打大频项集合并

#思路，生成1频项集，根据最小支持度过滤；依次由m频项集生成m+1频项集,直到m+1频项集为空停止迭代。
def apriori(dataSet,minSupport=0.5):
    C1 = createC1(dataSet) #创建1频项集
    D = list(map(set,dataSet)) #取每一条数据的集合（去除每一条数据的重复项）
    L1,supportData = scanD(D,C1,minSupport) #返回满足最小支持度的候选集和所有候选集项及对应的支持度
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0): #创建包含更大项集的更大列表,直到下一个大的项集为空
        Ck= aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData

"""
#测试
dataSet = loadDataSet()
L,supportData = apriori(dataSet)
print(L) #[[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})], [frozenset({2, 3}), 
# frozenset({3, 5}), frozenset({2, 5}), frozenset({1, 3})], [frozenset({2, 3, 5})], []]
print(supportData)
C2 = aprioriGen(L[0],2)
print(C2)
C3 = aprioriGen(L[1],3)
print(C3)
"""

#计算后件为H的规则的置信度，代码可以看出只是一个条件概率公式而已；根据最小置信度，筛选出有趣的规则；
#calcConf()函数是二频项集的关联规则，也就是H只能有一个元素
def calcConf(freqSet,H,supportData,brl,minConf=0.7):#频繁项集列表，出现在规则右部的H
    prunedH = []
    for conseq in H: #这里的conseq是单个元素
        conf = supportData[freqSet] / supportData[freqSet-conseq]
        if conf >= minConf:
            #print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq,conseq,conf)) #添加到规则里，brl 是后面通过检查的 bigRuleList
            prunedH.append(conseq)
    return prunedH

#rulesFromConseq()函数是如果频繁项集的元素数目超过1，那么就考虑做进一步的合并，函数rulesFromConseq()完成合并过程
def rulesFromConseq(freqSet,H,supportdata,brl,minConf=0.7): #H是可以出现在规则右部的元素列表
    #print(H) #[frozenset({2}), frozenset({3}), frozenset({5})]
    m = len(H[0])
    #print(m)
    if len(freqSet) > (m+1): #频繁项集是否大道可以移除大小为m的子集，如果可以，则移除
        Hmp1 = aprioriGen(H,m+1) #aprioriGen生成H中元素的无重复组合，Hmp1下一次迭代的H列表
        #由后件数为m的规则集生成后件为后件数为m+1的规则集，并计算置信度；
        # 递归到没有可以合并的规则集停止
        Hmp1 = calcConf(freqSet,Hmp1,supportdata,brl,minConf=0.7)
        if (len((Hmp1)) > 1): #如果不止一条规则满足要求，则使用rulesFromConseq()来判断是否可以进一步组合这些规则
            rulesFromConseq(fre ,Hmp1,supportdata,brl,minConf)

#思路，由于规则的前后件均不能为空，所以只有二频繁项集才能产生关联规则；

#1）首先由二频繁项集生成规则集，遍历所有的二频繁项集（每个元素轮流作为后件），根据最小置信度过滤规则集；

#       eg 二频繁项集{12},则计算规则{1}->{2}和{2}->{1}的置信度；

#2）依次迭代，在三大频项集生成规则集（每个元素轮流作为后件），需要考虑规则的合并，

#       eg 三大频项集{123},则{12}->{3},{13}->{2},{23}->{1}，此外考虑合并，{1}->{23},{2}->{13},{3}->{12},
#           还要继续合并：根据后件，前k-2个同的合并，本例前k-2个同的个数为0，所以停止

#这里的生产成规则后件都是一个元素
def generateRules(L,supportdata,minConf=0.7):
    bigRuleList = [] #存储包含可信度的规则列表
    for i in range(1,len(L)):#从1开始，即第二个元素开始，L(0)是单个元素(不能产生关联规则)
        for freqSet in L[i]: #频繁项集某一项中遍历每一个元素
            H1 = [frozenset([item]) for item in freqSet] #H1只包含单个元素
            if i > 1: #如果频繁项集的元素数目超过1，那么就考虑做进一步的合并，函数rulesFromConseq()完成合并过程
                rulesFromConseq(freqSet,H1,supportdata,bigRuleList,minConf)
            else: #如果频繁项集的元素个数为1，则使用calcConf计算可信度
                calcConf(freqSet,H1,supportdata,bigRuleList,minConf)
    return bigRuleList

"""
#测试
dataSet = loadDataSet()
L,supportData = apriori(dataSet,minSupport=0.5)
rules = generateRules(L,supportData,minConf=0.5)
print(rules)
"""

"""
#示例：发现毒蘑菇相似的特征
mushdata = [line.split() for line in open(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\'
                                         r'machinelearninginaction\\Ch11\\mushroom.dat').readlines()]
L,supportdata = apriori(mushdata,minSupport=0.3)
for item in L[1]:
    if item.intersection('2'): #交集，item中包含2的
        print(item)
"""

#scikit-learn中无对应的包


