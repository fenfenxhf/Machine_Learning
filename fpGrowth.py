from numpy import *

#创建一个类来保存树的每一个节点
class treeNode(object):
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue #节点名称
        self.count = numOccur #存放计数值
        self.nodeLink = None #链接相似元素
        self.parent = parentNode #存放父节点
        self.children = {} #存放子节点

    def inc(self,numOccur):
        self.count += numOccur

    #展示树的结构
    def disp(self,ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1) #采用递归的方式对不同层级的节点输出不同个数的空格

"""
#测试
rootNode = treeNode('pyramid',9,None)
rootNode.children['eye'] = treeNode('eyegdssg',13,None)
rootNode.children['ear'] = treeNode('eargdssg',2,None)
rootNode.disp()
"""


#构建FP树

def updateHeader(nodeToTest,targetNode):
    while nodeToTest.nodeLink != None: ##迭代指针链表，直到nodeToTest的指针为空
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode #将targetNode加入指针链表

def updateTree(items,inTree,headerTable,count):#items是排序后的数据集；inTree是树结构（调用了treeNode类）；
    #headerTable是头指针，count是每条数据出现的次数
    #print(items)
    #print(headerTable)
    if items[0] in inTree.children: #如果items[0]项(第一个元素)是否作为子节点,那么增加树种对应节点计数值
        #print(items[0])
        inTree.children[items[0]].inc(count) #inc（）增加计数
    else:
        inTree.children[items[0]] = treeNode(items[0],count,inTree) #如果items[0]项不在树中,那么增加树的节点，创建分支
        #这时头指针表也要更新以指向新的节点
        if headerTable[items[0]][1] == None:#如果头指针为空，更新headerTable头指针
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:#如果items[0]对应的头指针非空,调用updateHeader()更新头指针
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::],inTree.children[items[0]],headerTable,count)

def createTree(dataSet,minSup=1): #数据集，最小支持度
    headerTable = {} #第一次遍历数据集并统计每个元素出现的频度，保存在头指针表中
    for trans in dataSet:#第一次遍历数据集
        for item in trans:#每一条数据的每一个元素
            headerTable[item] = headerTable.get(item,0) + dataSet[trans]
            #headerTable[item] = headerTable.get(item, 0) + 1  #和上面的效果一样（只是这个数据集一样）

        #print(dataSet[trans]) #dataSet是以字典的形式传递，dataSet[trans]就是值，也就是这条记录出现的次数，不一定是1
    #print(headerTable) #每个元素和出现的频度

    #for k in headerTable.keys(): #遍历每一个元素
        #if headerTable[k] < minSup:
            #del headerTable[k] #移除不满足最小支持度的元素项
    headerTable = {k: v for k, v in headerTable.items() if v >= minSup} #python3中这一行代码代替上面三行代码，否则报错
    #print(headerTable) #结果是：{'r': 3, 'z': 5, 't': 3, 'x': 4, 'y': 3, 's': 3}

    freqItemSet = set(headerTable.keys()) #保存达到要求的数据

    if len(freqItemSet) == 0: #如果没有元素项达到最小支持度则退出
        return None,None

    for k in headerTable:#遍历头指针，改变头指针每个键所对应的值
        headerTable[k] = [headerTable[k],None] #保存计数值及指向每种类型第一个元素项的指针
    #print(headerTable) #结果是：{'r': [3, None], 'z': [5, None], 't': [3, None], 'x': [4, None], 's': [3, None], 'y': [3, None]}

    retTree = treeNode('Null Set',1,None) #创建只包含空集合的根结点

    #下面的代码片完成书上表12-2的转化，因为每个事物都是无序集合如{z,x,y}和{y,z,x}其实是同一个项，所以需要排序
    for tranSet,count in dataSet.items():##对数据集的每一条记录
        #  第二次遍历，只考虑频繁项,dataSet.items()返回字典的键值对(每条记录和该记录出现的次数)
        localD = {}#localD存储tranSet中满足最小支持度的项和对应的出现频率
        for item in tranSet: #字典的键中的每个值，也就是每一条数据记录每一个出现的元素
            #print(item)
            if item in freqItemSet: #如果这个元素在频繁数据中
                localD[item] = headerTable[item][0] #取列表的第一位（出现的次数），第二位是None
                #print(headerTable[item][0])
        #print(localD)
        # 使用排序后的频率项集对树进行填充
        if len(localD) > 0:
            orderItems = [v[0] for v in sorted(localD.items(),key=lambda p: p[1], reverse=True)]
            #对每一条记录进行排序，降序排序
            #print(orderItems)
            updateTree(orderItems,retTree,headerTable,count) #用排序后的数据集去更新树结构
    return retTree,headerTable #返回树结构和头指针

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

#createInitSet() 用于实现上述从列表到字典的类型转换过程
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

"""
#测试
simpDat = loadSimpDat()
print(simpDat)
initSet = createInitSet(simpDat)
print(initSet)
myFPtree,myHeaderTab = createTree(initSet,3)
#print(myHeaderTab)
myFPtree.disp()
"""

#发现以给定项结尾的所有路径的函数，一旦到达了某个元素项，就可以上溯这棵树直到根节点为止
#给定项结尾的计数值就是该条路径的计数值
def ascendTree(leafNode,prefixPath): #
    if leafNode.parent != None: #如果有父节点
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath) #迭代

def findPrefixPath(basePat,treeNode):
    condPats = {} #
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode,prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count #该节点往上就是路径，计算出路径出现的次数
        treeNode = treeNode.nodeLink
    return condPats

"""
simpDat = loadSimpDat()
#print(simpDat)
initSet = createInitSet(simpDat)
#print(initSet)
myFPtree,myHeaderTab = createTree(initSet,3)
#myFPtree.disp()
#print(myHeaderTab)
condPats = findPrefixPath('r',myHeaderTab['r'][1])
print(condPats)
"""

def mineTree(inTree,headerTable,minSup,preFix,freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(),key=lambda p:p[1][0])]

    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat,headerTable[basePat][1])
        myCondTree,myHead = createTree(condPattBases,minSup)
        if myHead != None:
            mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)
"""
initSet = createInitSet(simpDat)
#print(initSet)
myFPtree,myHeaderTab = createTree(initSet,3)
freqItems = []
mineTree(myFPtree,myHeaderTab,3,set([]),freqItems)
"""

"""
#示例：从新闻网站点击流中挖掘
parsedData = [line.split() for line in open(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\'
                                            r'machinelearninginaction\\Ch12\\kosarak.dat').readlines()]
initSet = createInitSet(parsedData)
myFPtree,myHeaderTab = createTree(initSet,100000) #寻找至少被100000人浏览过的新闻报道
myFreqList = []
mineTree(myFPtree,myHeaderTab,100000,set([]),myFreqList)
print(len(myFreqList))
print(myFreqList)
"""











