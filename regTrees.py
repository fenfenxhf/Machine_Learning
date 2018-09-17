#当数据非常复杂时候，构建全局模型就显得很难，可以构建多个模型
#回归树和分类树思路类似，但是叶节点的数据是连续型而不是离散型
#CART只做二元切分来处理连续型变量
from numpy import *
from tkinter import *
import matplotlib
matplotlib.use("TkAgg") #Agg可以从图像创建光栅图，实现绘图与应用之间的接口，把Agg呈现在画布上
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


#建立树节点
class treeNode():
    def __init__(self,feat,val,right,left):#待切分的特征；待切分的特征值；右子树（当不需要切分时也可以是单个值）；左子树
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left

#CART算法的实现
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)

    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #将每行映射成浮点数
        fltLine3 = list(fltLine)#python3要做一次修改，添加这一行，才能正确显示
        dataMat.append(fltLine3)

    #print(dataMat)
    return dataMat #返回数据集

#将数据集切分
def binSplitDataSet(dataSet,feature,value):
    #mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0],:][0] #将大于value的值放在左树
    #mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0],:][0] ##将小于等于value的值放在右树
    #书上有误，以下是正确代码
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]

    return mat0,mat1

"""
#测试
testMat = mat(eye(4))
mat0,mat1 = binSplitDataSet(testMat,1,0.5)
print(mat0)
print(mat1)
"""

#回归树叶节点返回所有数据的均值
def regLeaf(dataSet):
    return mean(dataSet[:,-1]) #最后一列标签列的均值

#返回数据集的误差：为了成功构建以分段常数为叶节点的树，需要度量出数据的一致性，计算连续型数值的混乱度，可以用总方差来代替
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0] #这里用平均误差的总值（总方差）


def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    #数据集；回归模型（回归树还是模型树）；误差类型；ops包含树构建所需其他参数的元组
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops) #在chooseBestSplit函数中返回
    if feat == None: #满足条件时返回叶节点的值，其中None是chooseBestSplit函数中的返回值
        return val
    retTree = {} #保存树的结构信息
    retTree["spInd"] = feat
    retTree["spVal"] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val) #在binSplitDataSet函数中返回左右子树
    retTree["left"] = createTree(lSet,leafType,errType,ops) #递归创建树
    retTree["right"] = createTree(rSet, leafType, errType, ops)
    #print(retTree)
    return retTree


#选择最好的划分特征和特征值
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0];tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #数据集最后一列标签值都相等时
        return None,leafType(dataSet) #提前终止切分，并返回叶节点标签的均值

    m,n = shape(dataSet)
    S = errType(dataSet) #计算未切分前数据集的误差
    bestS = inf;bestIndex = 0;bestValue = 0 #初始化

    for featIndex in range(n-1): #n-1个特征，最后一列是标签，遍历每一个特征

        #for splitVal in set(dataSet[:,featIndex]): #对每一个特征值,将数据切成两份， #python3报错，修改如下
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]): #对每一个特征值
            mat0, mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue #当叶节点数据个数小于tolN个时候，就提前终止切分
            # #提前终止循环是预剪枝，但是对tolS和tolN很敏感
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: #如果切分后的误差又减小了
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    if (S - bestS) < tolS: #如果切分后两次误差的差值小于tolS,则相当于误差不大，所以没必要再切分
        return None,leafType(dataSet)

    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): #检查最后切分的mat0,mat1是否满足不切分的条件
        return None,leafType(dataSet)
    return bestIndex,bestValue

"""
#测试
myDat = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch09\\ex00.txt')
myMat = mat(myDat)
#print(myMat)
regtree = createTree(myMat)
print(regtree)

#测试数据对ops的敏感度
myDat2 = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch09\\ex2.txt')
myMat2 = mat(myDat2)
#print(myMat)
regtree = createTree(myMat2,ops=(500,4))
print(regtree)
"""

#回归树剪枝：后剪枝。需要将数据集分为测试集和训练集，用测试集将当前两个叶节点合并后误差与不合并误差相比较，
# 如果误差降低，则将叶节点合并
def isTree(obj):
    return type(obj).__name__ == 'dict' #返回True or Flase

#获得左右节点的平均值
def getMean(tree):
    if isTree(tree['right']): #如果右子树不是叶节点而是下面还有子树
        tree['right'] = getMean(tree['right']) #则递归调用getMean()直到叶节点
    if isTree(tree['left']): #如果左子树不是叶节点而是下面还有子树
        tree['left'] = getMean(tree['left']) #则递归调用getMean()直到叶节点
    return (tree['right'] + tree['left']) / 2.0

def prune(tree,testData):
    if shape(testData)[0] == 0: #测数据没有数据，对树进行塌陷处理
        return getMean(tree)

    #根据训练集训练的树切分测试集
    if isTree(tree["right"]) or isTree(tree["left"]): #如果左右树还有子树
        lSet,rSet = binSplitDataSet(testData,tree["spInd"],tree["spVal"]) #对测试集进行切分
    if isTree(tree["left"]): #如果左子树下面还有子树
        tree["left"] = prune(tree["left"],lSet) #则将测试集的lSet递归切分
    if isTree(tree["right"]):#如果右子树下面还有子树
        tree["right"] = prune(tree["right"], rSet)#则将测试集的rSet递归切分

    if not isTree(tree["right"]) and not isTree(tree["left"]):#如果左右已经没有子树
        lSet, rSet = binSplitDataSet(testData, tree["spInd"], tree["spVal"])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2)) #计算不合并的误差
        treeMean = (tree['right'] + tree['left']) / 2.0 #合并之后的取左右子节点的均值
        errorMerge = sum(power(testData[:,-1] - treeMean, 2)) #计算合并后的误差
        if errorMerge < errorNoMerge:
            return treeMean
        else:
            return tree
    else:
        return tree

"""
#测试
myDat2 = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch09\\ex2.txt')
myMat2 = mat(myDat2)
#print(myMat)
mytree = createTree(myMat2,ops=(0,1))
myDatTest = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch09\\ex2test.txt')
myMat2Test = mat(myDatTest)
tree = prune(mytree, myMat2Test)
print(tree)
"""

#模型树：不将叶节点设定为常数值，而是设置为分段函数
#叶节点生成线性函数
def linearSolve(dataSet): #用线性模型拟合叶节点数据
    m,n = shape(dataSet)
    X = mat(ones((m,n)));Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1] #第一列被初始化为1，剩下的为dataSet前n-1个特征
    #如果特征比较多，如何将第一列初始化为1，就是生成全1矩阵，然后第一列不动，剩下的列赋值数据
    Y = dataSet[:,-1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,try increasing the second value of ops')
    ws = xTx.I * X.T * Y
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

"""
#测试
myDat2 = loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch09\\exp2.txt')
myMat2 = mat(myDat2)
mytree = createTree(myMat2,modelLeaf,modelErr,(1,10))
print(mytree)
"""

#示例：树回归(回归树 模型树)和标准回归的比较
def regTreeEval(model,inDat): #回归树,为了和modelTreeEval（）函数保持一致，所以给了两个参数
    return float(model)   #返回一个数值

def modelTreeEval(model,inDat): #模型树
    n = shape(inDat)[1]
    X = mat(ones((1,n+1))) #第一列为1，最后一列是标签
    X[:,1:n+1] = inDat
    return float(X * model) #y=x*w

# 预测函数
# 类似二叉树递归遍历, 走到叶子节点后,
# 就调用 modelEvel 处理叶子节点, 得到的返回值即为预测值
def treeForecast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree): #如果是叶节点
        return modelEval(tree,inData) #返回这个节点的值，也就是均值

    if inData[tree["spInd"]] > tree["spVal"]: #输入值大于阈值，则放在左树
        if isTree(tree["left"]): #如果左树低下有子树
            return treeForecast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree["left"],inData)
    else: #输入值小于等于阈值，则放在右树
        if isTree(tree["right"]):
            return treeForecast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree["right"],inData)

def createForeCast(tree,testData,modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i, 0] = treeForecast(tree, mat(testData[i]), modelEval)
    return yHat


"""
#测试
#创建一颗回归树
trainMat = mat(loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction'
                           r'\\Ch09\\bikeSpeedVsIq_train.txt'))
testMat = mat(loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction'
                           r'\\Ch09\\bikeSpeedVsIq_test.txt'))
myTree = createTree(trainMat,ops=(1,20))
#print(myTree)
yHat = createForeCast(myTree,testMat[:,0]) #数据的第0列是特征
print(corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]) #[0,1]是因为corrcoef（）返回的是对角矩阵，主对角为1，副对角相等，所以取一个元素就好
#rowvar=1是默认，表示每行代表一个变量，每列代表一个样本；rowvar=0反之
#0.9640852318222151

#创新模型树
myTreeModel = createTree(trainMat,modelLeaf,modelErr,(1,20))
yHatModel = createForeCast(myTreeModel,testMat[:,0],modelTreeEval)
print(corrcoef(yHatModel,testMat[:,1],rowvar=0)[0,1]) #0.9760412191380607

#标准线性
ws,X,Y = linearSolve(trainMat)
#print(ws)
yHatLine = mat(ones((shape(testMat)[0],1)))
for i in range(shape(testMat)[0]):
    yHatLine[i,0] = testMat[i,0] * ws[1,0] + ws[0,0]
print(corrcoef(yHatLine,testMat[:,1],rowvar=0)[0,1])  #0.9434684235674765
"""

#使用python的tkinter库创建GUI（图形用户界面）
"""
#测试
myLabel = Label(root,text="Hello World!")
myLabel.grid()
root.mainloop()
"""

"""
#用于构建树管理器界面的tkinter小部件
def reDraw(tolS,tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get(): #切换模型树
        if tolN < 2: tolN = 2
        myTree=createTree(reDraw.rawDat, modelLeaf,modelErr, (tolS,tolN))
        yHat = createForeCast(myTree, reDraw.testDat, modelTreeEval)
    else: #回归树
        myTree=createTree(reDraw.rawDat, ops=(tolS,tolN))
        yHat = createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:,0].A, reDraw.rawDat[:,1].A, s=5) #use scatter for data set
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0) #use plot for yHat
    reDraw.canvas.show()

def getInputs(): #获得tolS,tolN
    try: tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0,'10')
    try: tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0,'1.0')
    return tolN,tolS

def drawNewTree():
    tolN,tolS = getInputs()
    reDraw(tolS,tolN)

root = Tk()

#Label(root,text="Plot Place Holder").grid(row=0,columnspan=3) #0行，跨3列
#先用画布来替代占位符，删除对应标签，添加以下代码
reDraw.f = Figure(figsize = (5,4),dpi = 100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row = 0,columnspan = 3)

Label(root,text="tolN").grid(row=1,column=0) #一行，0列
tolNentry = Entry(root) #Entry文本输入框
tolNentry.grid(row=1,column=1) #1行，1列
tolNentry.insert(0,'10')
Label(root,text="tolS").grid(row=2,column=0) #一行，0列
tolSentry = Entry(root) #Entry文本输入框
tolSentry.grid(row=2,column=1) #1行，1列
tolSentry.insert(0,'1.0')
Button(root,text="ReDraw",command=drawNewTree).grid(row=1,column=2,rowspan=3)

chkBtnVar = IntVar() #按钮整数值
chkBtn = Checkbutton(root,text='Model Tree',variable=chkBtnVar) #复选按钮
chkBtn.grid(row=3,column=0,columnspan=2)


reDraw.rawDat = mat(loadDataSet(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch09\\sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01) #最小值，最大值，步长

reDraw(1.0,10)

root.mainloop()
"""










