"""
#将txt文本转化成excel格式
#coding='utf-8'
import datetime
import time
import os
import sys
import xlwt

def txt2xls(filename,xlsname):
    f = open(filename)
    x = 0;y = 0 #在excel开始写的位置
    xls = xlwt.Workbook()
    sheet = xls.add_sheet('sheet1',cell_overwrite_ok=True)

    while True: #循环读取文本中所有的内容
        line = f.readline() #一行一行读取
        if not line: #如果这行为空，就是没有内容
            break #则跳出循环
        for i in line.strip().split('\t'): #读取这一行中相应的内容
            item = i.strip()
            sheet.write(x,y,item)
            y += 1 #另起一列
        x += 1 #另起一行
        y = 0 #另起一行时候列是从头开始写

    f.close()
    xls.save(xlsname+'.xls') #保存excel

txt2xls(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch03\\lenses.txt',
        r'C:\\Users\\Administrator\\Desktop\\lenses_xls')
"""

"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score #导入计算交叉验证值的函数
from sklearn.cross_validation import train_test_split #导入测试集和训练集划分函数
#from sklearn.datasets import load_iris

import numpy as np

#iris = load_iris()
#print(iris.data)

fr = open(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch03\\lenses.txt')
dataMat = []
#Labels = []
for line in fr.readlines():
    currLine = line.strip().split('\t')
    #print(currLine)
    #currArr = currLine[:-1]
    dataMat.append(currLine)
    #Labels.append(currLine[-1])
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
#X = np.array(dataMat)
#print(dataMat)
#print(X)
#print(len(dataMat))
#print(len(dataMat))
#print(lensesLabels)

#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = DecisionTreeClassifier(criterion='entropy')
#print(clf)
clf.fit(dataMat,lensesLabels)
#clf.fit(dataMat,Labels) #直接使用fit报错：ValueError: could not convert string to float: 'young'
"""

#  为了对string类型的数据序列化，需要先生成pandas数据，这样方便我们的序列化工作。
# 这里我使用的方法是，原始数据->字典->pandas数据
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn import tree
import pydotplus


if __name__ == '__main__':
    with open(r'E:\\奋斗历程\\python\\MLiA_SourceCode\\machinelearninginaction\\Ch03\\lenses.txt') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()] #将txt文件变成列表形式
    #print(lenses)
    lenses_target = [] #存储最好一列标签列
    for each in lenses:
        lenses_target.append(each[-1])
    #print(lenses_target)

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate'] #特征标签
    lenses_list = []
    lenses_dict = {}
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    #print(lenses_dict)
    lenses_pd = pd.DataFrame(lenses_dict) #转化成pandas数据
    #print(lenses_pd)

    le = LabelEncoder() #实例化对象，用于序列实例化
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    #print(lenses_pd)

    #使用Graphviz可视化决策树
    clf = DecisionTreeClassifier(max_depth=4) #实例化
    clf = clf.fit(lenses_pd.values.tolist(),lenses_target) #使用pandas数据构建决策树

    dot_data = StringIO() #实例化对象，StringIO经常用于字符串缓存
    tree.export_graphviz(clf,out_file=dot_data,feature_names=lenses_pd.keys(),
                         class_names=clf.classes_,filled=True,rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) #dot_data.getvalue()返回对象中所有的数据
    graph.write_pdf(r'C:\\Users\\Administrator\\Desktop\\tree.pdf')

    pre = clf.predict([[1,1,1,0]]) #测试
    #print(pre)






