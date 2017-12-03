# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:52:29 2017

@author: Q
"""

import numpy as np
def loadDataSet(fileName):  
    dataMat = [] 
    with open(fileName) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            dataMat.append(line)
    dataMat = np.array(dataMat).astype(np.float32)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]  #每个特征可以被多次划分
    mat1 = dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1
    
def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])
def regErr(dataSet):
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0]#容许的误差下降值
    tolN = ops[1]#切分后的最小样本数
    dataMat = np.mat(dataSet)
    n,m = np.shape(dataMat)
    featureFlag = 0
    for i in range(m-1):   #检查一下是否所有特征都相等
        if len(set(dataMat[:,i].T.tolist()[0])) !=1:
            featureFlag = 1
    if featureFlag == 0:
        return None,leafType(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featureIndex in range(m-1):
        for featureValue in set(dataMat[:,featureIndex].T.tolist()[0]):
            mat0,mat1 = binSplitDataSet(dataSet,featureIndex,featureValue)
            if(np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featureIndex
                bestValue = featureValue
    if (S - bestS) < tolS:
        return None,leafType(dataMat)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if(np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN):   #最后还要检查分割子集是否过小
        return None,leafType(dataMat)
    return bestIndex,bestValue

    
def creatTree(dataSet,leafType = regLeaf,errType = regErr,ops = (0,1)):
    feat,value = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:
        return value
    retTree  = {}
    retTree['spInd'] = feat
    retTree['spVal'] = value
    lSet,rSet = binSplitDataSet(dataSet,feat,value)
    retTree['left'] = creatTree(lSet,leafType,errType,ops)
    retTree['right'] = creatTree(rSet,leafType,errType,ops)
    return retTree
def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2
            
def prune(tree,testData):
    if np.shape(testData)[0] == 0:#如果没有数据，则对树进行i塌陷处理
        return getMean(tree)
    if isTree(tree['left']) or isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(np.power(rSet[:,-1] - tree['right'],2)) + sum(np.power(lSet[:,-1] - tree['left'],2))
        treeMean = (tree['right'] + tree['left'])/2
        errorMerge = sum(np.power(testData[:,-1]-treeMean,2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    return tree
        
def linearSolve(dataSet):
    m,n = np.shape(dataSet)
    dataMat = np.mat(dataSet)
    X = np.mat(np.ones((m,n)))
    X[:,1:n] = dataMat[:,0:n-1]#.copy()
    Y = dataMat[:,-1]#.copy()
    xTx = X.T*X
    if np.linalg.det(xTx) == 0:
        raise NameError('This matrix is singular')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    predictY = X * ws
    err = sum(np.power((predictY-Y),2))
    return err
    
    
def regTreeEval(model,inData):
    return float(model)
def modelTreeEval(model,inData):
    n = np.shape(inData)[1]
    X = np.mat(np.ones(1,n+1))
    X[:,1:n+1] = inData
    return float(X*model)

def treeForeCast(tree,inData,modelEval = regTreeEval):
    if not isTree(tree):
        return modelEval(tree,inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)
#data = loadDataSet('bikeSpeedVsIq_train.txt')
data = loadDataSet('exp2.txt')
myMat = np.mat(data)
tree1 = creatTree(myMat,ops=(1,10))
c = treeForeCast(tree1,[0.411198])
#data2 = loadDataSet('ex2test.txt')
#myMat2 = np.mat(data2)
#tree = prune(tree1,myMat2)
print(c)













