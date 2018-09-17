#####3-2#####
#使用文本注解绘制树节点
import matplotlib.pyplot as plt

# 引入中文字体
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

decisionNode = dict(boxstyle='sawtooth',fc='0.8') #boxstyle:文本框类型，sawtooth:锯齿形，fc:边框线条粗细
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle='<-') #定义箭头属性

def plotNode(nodeTxt,centerPt,parentPt,nodeType): #4个参数分别表示：要显示的文本，文本中心点，箭头线开始的坐标，属性
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
# 说明：annotate是注解工具，其中，xy是箭头尖的坐标，xytext设置注释内容显示的中心位置，
# xycoords和textcoords是坐标xy与xytext的说明（按轴坐标），若textcoords=None，则默认textcoords与xycoords相同，若都未设置，默认为data
# va/ha设置节点框中文字的位置，va为纵向取值为(u'top', u'bottom', u'center', u'baseline')，ha为横向取值为(u'center', u'right', u'left')

def createPlot(inTree):
    fig = plt.figure(1,facecolor='white') #创建一个画布，背景为白色
    fig.clf() #清空画布
    axprops = dict(xticks=[],yticks=[]) #定义横纵坐标
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops) # ax1是函数createPlot的一个属性，这个可以在函数里面定义
    # 也可以在函数定义后加入也可以     #frameon表示是否绘制坐标轴矩形 ，也就是外面的大框框 #**axprops无坐标轴
    #plotNode(u'决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
    #plotNode(u'叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

#createPlot()

#获得叶节点的数目和树的层数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0] #获得第一个键,myTree.keys()不能索引，所以转化成list
    secondDict = myTree[firstStr] #获得上面键的值，可能也是个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    #print(maxDepth) #测试getTreeDepth()调用了几次
    return maxDepth

#为了节省时间，函数retrieveTree输出预先存储的树信息，避免对此建立树的麻烦
def retrieveTree(i):
    listOfTree = [{'no surfacing':{ 0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
                  {'no surfacing':{ 0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}},
                  {'tearRate': {'reduced': 'no lenses', 'normal': {'astigmatic': {'no': {
                      'age': {'presbyopic': {'prescript': {'myope': 'no lenses', 'hyper': 'soft'}}, 'young': 'soft',
                              'pre': 'soft'}}, 'yes': {'prescript': {'myope': 'hard', 'hyper': {
                      'age': {'presbyopic': 'no lenses', 'young': 'hard', 'pre': 'no lenses'}}}}}}}}]
    return listOfTree[i]

#myTree = retrieveTree(0)
#print(myTree)
#numLeafs = getNumLeafs(myTree)
#numTreeDepth = getTreeDepth(myTree)
#print(numLeafs)
#print(numTreeDepth)

###绘制树
#在父子节点间填充文本信息
def plotMidText(cntrPt,parentPt,txtString): #cntrPt子文本中心点坐标，parentPt箭头开始的坐标，txtString要填充的文本信息
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 +cntrPt[0] #x中间点
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1] #y中间点
    createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt): #nodeTxt节点要填充的文本
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yoff) #计算文本中心点
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict': #如果还是字典类型，则就还有树结构
            plotTree(secondDict[key],cntrPt,str(key)) #递归绘制树，secondDict[key]是子树传递给myTree，子文本中心点传递给箭头开始点，key转化成str格式传递给需要填充的文本
        else:
            plotTree.xoff = plotTree.xoff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xoff,plotTree.yoff),cntrPt,leafNode)
            plotMidText((plotTree.xoff,plotTree.yoff),cntrPt,str(key))
    plotTree.yoff = plotTree.yoff + 1.0/plotTree.totalD

#myTree = retrieveTree(2)
#createPlot(myTree)




