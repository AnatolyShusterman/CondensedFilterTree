'''
This code was written by:
Anatoly Shusterman
Tal Ivshin
'''

import itertools
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin,clone
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import f1_score
from sklearn.metrics import label_ranking_loss

'''
Class Node:
This class represents a node in the tree.

Attributes:
-classifier:the classifier that was buid in the node
-labels: 
-right: the right child
-left: the left child
-sebiling;
'''
class Node():
    def __init__(self,classifier=None,labels=None):
        self.classifier=classifier
        self.labels=labels
        self.right=None
        self.left=None
        self.sebiling=None

'''
Class NothingClassifier:
This class is used in the leaves, which dont contain any real classifier
'''
class NothingClassifier(ClassifierMixin):
    def fit(self,X,y):
        self.consty=y
    def predict(self,X):
        return [self.consty]*len(X)

'''
Class CondensedFilterTreeClassifier inherits from ClassifierMixin:
This class represents the CFT algorithm

Attributes:
-_binClassifier: the classifier that each node in the tree will contain
-_tree: this parameter hold the tree structer of the classifier(points to the root)
-M: number of iterations
-_treeDict:
-_arrDict:
-_loss: the parameter hold the name of the loss function to be used
-_classCount:
-_labelToInt:
-_intToLabel:
-
'''
class CondensedFilterTreeClassifier(ClassifierMixin):

    '''
    Constructer
    '''
    def __init__(self):
        self._tree=None
        self._treeDict = dict() #string->node
        self._arrDict=dict() #string->array
        self._classCount=0
        self._labelToInt=dict()
        self._intToLabel=dict()
        self._nodeSplitDict=dict()

    def set_params(self,m=3,loss='hamming',classifier = None):
        if classifier == None:
            self._binClassifier = GaussianNB()
        else:
            self._binClassifier = classifier

        self._M = m
        self._loss = loss

    def get_params(self):
        return self._M,self._loss,self._binClassifier

    def _getNewBinClassifier(self):
        return clone(self._binClassifier)

    def _fillTreeDict(self, y):
        n=np.size(y,1)
        self._classCount=0
        vals=[list(i) for i in itertools.product([0, 1], repeat=n)]
        tab = str.maketrans(dict.fromkeys('[],\x20'))
        keys= [str(x).translate(tab) for x in vals]
        self._arrDict=dict()
        for i,k in enumerate( keys):
            self._arrDict[k]=vals[i]
            self._labelToInt[k]=int(k,2)
            self._intToLabel[int(k,2)]=k
            self._classCount = self._classCount +1
    '''
    This function represents the cost function in the algorithm.
    Parameters:
    -y_ni: real value class
    -t_ni: predicted value class
    '''
    def _costFunction(self,y_ni,t_ni):
        res=0.0
        if self._loss == 'hamming':
            res = hamming_loss(y_ni, t_ni)
        elif self._loss == 'rank':
            res=label_ranking_loss(y_ni, t_ni)
        elif self._loss == 'f1':
            res=1-f1_score(y_ni, t_ni, average='binary')
        return res

    '''
    Function for training each node in the CFT tree
    Parameters:
    -Xn: observations and attributes
    -yp: predicted class
    -tn:
    -y: readl class value
    '''
    def _trainNode(self,Xn,yp,tn,y):
        Dk = pd.DataFrame();bn = [];wn = []
        for i,x in Xn.iterrows():
            t=self._arrDict[yp[i]]# is the previous iteration prediction
            Zn=x.tolist()+ t
            tl,tr=self._splitNodeVector(tn)
            c0=self._costFunction(list(map(int,y[i])),self._arrDict[tl])
            c1=self._costFunction(list(map(int,y[i])),self._arrDict [tr])
            if c0 < c1:
                bn.append(tl)
            else:
                bn.append(tr)
            wn.append(int(abs(c0 - c1)*1000))
            row=pd.DataFrame(data=[Zn])
            Dk=pd.concat([Dk,row],ignore_index=True)
        hk=self._getNewBinClassifier().fit(Xn, bn, sample_weight=np.asarray(wn))
        return hk

    '''
    Function used to split the node vector to left child and right child
    Parameters:
    -tn: current node value to be splitted
    '''
    def _splitNodeVector(self,tn):# convert 11111 to 11000 and 00111
        if tn in self._nodeSplitDict:
            return self._nodeSplitDict[tn][0],self._nodeSplitDict[tn][1]
        tl=""
        tr=""
        avr=tn.count('1')/2
        sm=0
        for l in tn:
            sm=sm+int(l)
            if sm <=avr:
                tl=tl+l
                tr=tr+'0'
            else:
                tl=tl+'0'
                tr=tr+l
        self._nodeSplitDict[tn]=(tl,tr)
        return tl,tr

    '''
    Function for CFT model training
    Parameters:
    -X: observations and attributes (table)
    -y: real class value (vector of 0/1)
    '''
    def fit(self,X,y):
        self._X=pd.DataFrame(X)
        self._fillTreeDict(y)
        y=pd.DataFrame(y)
        m=0
        ym=y.values.tolist()
        tab=str.maketrans(dict.fromkeys('[],\x20'))
        self._y = [str(x).translate(tab) for x in ym]
        ym=self._y.copy()
        ypm = ym.copy() #predicted y for iteration m
        Xm=self._X.copy() # X for iteration m
        self._tree = self._fitRecursive(Xm, ypm,'1'*len(self._y[0]),ym)
        m=m+1
        while m < self._M:
            yp = self._iterPredict(self._X)
            ypm = ypm + yp
            Xm = pd.concat([Xm, self._X], ignore_index=True)
            ym=ym+ self._y
            self._tree = self._fitRecursive(Xm, ypm,'1'*len(self._y[0]) ,ym)
            m=m+1

    '''
    Function for CFT tree training
    Parameters:
    -Xn: observations and attributes (table)
    -yp: predicted class(vector of 0/1)
    -tn:
    -y: real class value (vector of 0/1)
    '''
    def _fitRecursive(self, Xn, yp, tn, y): #t=[~yn[1],~yn[2]...~yn[k]]
        if tn.count('1') == 1:
            hk= NothingClassifier()
            hk.fit(Xn, tn) #always returns classification of tn
            n = Node(classifier=hk, labels=tn)
            self._treeDict[tn] = n
            return n
        else:
            tl, tr = self._splitNodeVector(tn)
            n=Node()
            n.classifier=self._trainNode(Xn, yp, tn, y)
            n.left = self._fitRecursive(Xn, yp,tl, y)
            n.right = self._fitRecursive(Xn, yp,tr ,y)
            n.left.sebiling=n.right
            n.right.sebiling=n.left
            n.labels=tn
            self._treeDict[tn]=n
        return n
    '''
    Function for observations predictions. works in iterative mode.
    Parameters:
    -X: observations and attributes(table)
    '''
    def _iterPredict(self, X):
        res=[]
        for index,row in X.iterrows():
            res.append(self._predictOne(row))
        return res

    '''
    Function for making prediction for 1 observation
    Parameters:
    -Xi: single observation and attributes(row)
    '''
    def _predictOne(self,Xi):

        Xi=pd.DataFrame(data= [Xi.tolist()])
        prediction= self._recursivePredictOne(Xi,(0,0),self._tree)
        return self._intToLabel[ prediction[0]]

    '''
    This function used to predict class values for a single observation in a recursive fashion.
    It goes over the tree and builds the predicted vector value
    Parameters:
    -Xi: single observation and attributes(row)
    node: current node that contains model that used for prediction
    '''
    def _recursivePredictOne(self,Xi, prediction,node):
        if node.labels.count('1') != 1:
            pp=node.classifier.predict_proba(Xi)[0]
            labels=node.classifier.classes_
            decision=0
            if pp[0] < pp[1] :
                decision=1
            if pp[decision] >prediction[1]:
                prediction=(self._labelToInt[labels[0]],pp[decision])
            prediction= self._recursivePredictOne(Xi,prediction,self._treeDict[labels[decision]])
        return prediction
    '''
    Genral CFT predict function (uses the _iterpredict(X))
    Parameters:
    -X:observations and attributes(table)
    '''
    def predict(self,X):
        X=pd.DataFrame(X)
        st=self._iterPredict(X)
        res = [self._arrDict[y] for y in st]
        return res