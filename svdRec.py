from numpy import *

#Data = U * sigma * V   m*n = (m*m) * (m*n) * (n*n)
#其中sigma只有对角元素，其他元素均为0，sigma的对角元素从大到小排列，
# 这些对角元素是奇异值，对应原始数据的奇异值，也是Data*data.T特征值的平方根
#在某个奇异值数目（r个）之后，其他的奇异值都为0，这意味着数据集中仅有r个重要特征，其余的都是噪声或冗余特征
def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]

"""
#测试
data = loadExData()
U,Sigma,VT = linalg.svd(data) #直接可以求矩阵的分解
#print(U)
print(Sigma) #Sigma是以行向量的形式返回，而不是对角矩阵的形式
#print(VT)
Sig3 = mat([[Sigma[0],0,0],[0, Sigma[1], 0],[0, 0, Sigma[2]]])#这里Sigma前三个数是非零，后面都是0
#print(Sig3)
rData = U[:,:3] * Sig3 * VT[:3,:]
print(rData)
"""

#相似度计算
#采用与欧式距离相关的计算公式
def euclidSim(inA,inB): #inA,inB都是列向量
    return 1.0 / (1.0 + linalg.norm(inA-inB)) #linlg,norm计算2范数，这样啊就是inA,inB之间的距离
    #如[3,4,5]的2范数为sqr(9+16+25)

#皮尔逊相关系数
def pearsSim(inA,inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5*corrcoef(inA,inB,rowvar=0)[0][1] #相关系数的值在[-1,1]之间，返回值是将范围变成0-1

#余弦相似度
def cosSim(inA,inB):
    num = float(inA.T * inB)
    denom = linalg.norm(inA) * linalg.norm(inB)
    return 0.5 + 0.5*(num/denom) #余弦值求出来也是在[-1,1]之间

"""
#测试
myMat = mat(loadExData())
es = euclidSim(myMat[:,0],myMat[:,4])
print(es)
ps = pearsSim(myMat[:,0],myMat[:,4])
print(ps)
cs = cosSim(myMat[:,0],myMat[:,4])
print(cs)
"""

#基于物品的相似度的推荐系统
def standEst(dataMat,user,simMeas,item): #用来计算在给定相似度计算方法的条件下，用户对物品的估计评分值
    #数据矩阵；用户编号；相似度计算方法；物品编号。数据矩阵中行表示每个用户，列表示每个物品
    #print(item)
    n = shape(dataMat)[1] #物品件数
    simTotal = 0.0;ratSimTotal = 0.0
    for j in range(n): #对每一件物品
        userRating = dataMat[user,j] #第user个用户对第j个物品的评分
        if userRating == 0:
            continue #遍历物品中，如果用户评分为0，则跳过这个物品
        overLap = nonzero(logical_and(dataMat[:,item].A > 0,dataMat[:,j].A > 0))[0]
        #寻找同时评价了item和j物品的两个用户，计算这两个用户在item,j下的相似度，根据j的评分和相似度得到对item的评分
        #print(overLap)
        #print(logical_and(dataMat[:,item].A > 0,dataMat[:,j].A > 0))
        if len(overLap) == 0: #如果用户没有评级任何物品
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
        #print('the %d and %d similarity is: %f'%(item,j,similarity))
        #print(dataMat[overLap,item])
        #print(dataMat[overLap,j])
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1] #寻找未评级的物品
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimateScore = estMethod(dataMat,user,simMeas,item) #调用函数返回未评分物品的估分
        itemScores.append((item,estimateScore))
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N] #评分逆序，返回最高的N个物品

"""
#测试
myMat = mat(loadExData())
myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
myMat[3,3]=2
print(myMat)
Nrecommendcs = recommend(myMat,2)
#print(Nrecommendcs)
#Nrecommendes = recommend(myMat,2,simMeas=euclidSim)
#print(Nrecommendes)
"""

#实际矩阵中会更加稀疏
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

"""
#测试
U,Sigma,VT = linalg.svd(mat(loadExData2()))
print(Sigma)
Sig2 = Sigma ** 2
ESig2 = sum(Sig2)
ESig2_09 = ESig2*0.9
print(ESig2_09)
print(sum(Sig2[:2]))
print(sum(Sig2[:3]))  #最后结果，可把11维的矩阵降成3维
"""

#基于SVD的评分估计
def svdEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0;ratSimTotal = 0.0
    U,Sigma,VT = linalg.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4]) #建立对角矩阵
    xformedItems = dataMat.T * U[:,:4] * Sig4.I #利用U矩阵降物品转化到低维空间中
    #print(xformedItems)

    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

"""
#测试
myMat = mat(loadExData2())
recommend2 = recommend(myMat,1,estMethod=svdEst)
print(recommend2) 
"""

#示例：基于SVD的图像搜索
def printMat(inMat,thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh: #如果大于阈值，则打印1
                print(1,)
            else:
                print(0,)
        print(' ')

def imgCompress(numSV=3,thresh=0.8):
    my1 = []
    for line in open(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch14\\0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        my1.append(newRow) #将32*32转化成1*1024 ，这里只转化了一张图片
    myMat = mat(my1)
    print("************************original matrix******************************")
    printMat(myMat,thresh)
    U,Sigma,VT = linalg.svd(myMat)
    SigRecon = mat(zeros((numSV,numSV)))
    for k in range(numSV):
        SigRecon[k,k] = Sigma[k] #创建对角矩阵
    reconMat = U[:,:numSV] * SigRecon * VT[:numSV,:]
    print("******reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat,thresh)

"""
#测试
imgCompress(2)
"""

"""
#以下用sklearn实现
from sklearn.decomposition import TruncatedSVD

#获得数据
my1 = []
for line in open(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch14\\0_5.txt').readlines():
    newRow = []
    for i in range(32):
        newRow.append(int(line[i]))
    my1.append(newRow) #将32*32转化成1*1024 ，这里只转化了一张图片
myMat = mat(my1)

clf = TruncatedSVD(n_components=2)

clf.fit(myMat)

newData = clf.fit_transform(myMat)

print(newData)
"""