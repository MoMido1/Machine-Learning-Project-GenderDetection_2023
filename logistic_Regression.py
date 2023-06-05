# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 13:45:37 2022

@author: Mohamed KH
"""
import sklearn.datasets
import numpy
import scipy.optimize
from numpy.linalg import norm


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

# D, L = 
# (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)


def center_data (dtr):
    mu = numpy.mean(dtr,axis = 1).reshape([dtr.shape[0],1])
    # print(mu)
    return dtr-mu
def standardize_variance(dtr):
    std = numpy.std(dtr,axis=1).reshape ([dtr.shape[0],1])
    return dtr/std
# def whiten_cov (dtr):
#     cov = numpy.cov(dtr)
#     A = numpy.linalg.matrix_power(cov,0.5)
#     return numpy.dot(A,dtr)
# print(DTR[:,0:5])
# print(center_data(DTR)[:,0:5])
def logreg_obj_wrap(dtr,ltr,prior,lmda):
    def logreg_obj (v):
       w,b = v[0:-1] , v[-1]
       # n = ltr.shape[0]
       # sumation=0
       dtp=dtr[:,ltr==1] 
       dtn=dtr[:,ltr==0]
       
       logexpp= numpy.logaddexp(0,-(numpy.dot(w.T,dtp)+b))
       logexpn= numpy.logaddexp(0,(numpy.dot(w.T,dtn)+b))
       
       sumationp = numpy.sum(logexpp)
       sumationn = numpy.sum(logexpn)
       
       J = lmda/2 * norm(w)**2 + prior/dtn.shape[1] * (sumationp) + (1-prior)/dtp.shape[1]* sumationn
       
       # for i in range(n):
       #     z =1 if ltr[i] else -1 
       #     x=dtr[:,i]
       #     logexp = numpy.logaddexp(0,-z*(numpy.dot(w.T, x)+b))
       #     sumation += logexp
       
       # J = lmda/2 * norm(w)**2 +1/n * (sumation)
       return J
    return logreg_obj

# DTR = center_data(DTR)
# DTR = standardize_variance(DTR)
# DTR = whiten_cov(DTR)


def train (DTR , LTR, args):
    prior = args[0]
    l = args[1]
#assuming a value of lambda to be 1 at first
    x0 =numpy.zeros(DTR.shape[0]+1)
    mybounds = [(-1000,1000) for _ in range(DTR.shape[0]+1)]
    logreg_obj = logreg_obj_wrap(DTR, LTR,prior, l)
    x,f,d=scipy.optimize.fmin_l_bfgs_b(logreg_obj,x0,approx_grad=True,iprint=1,bounds= mybounds)
    # print(f)
    # here x contains 5 values the first 4 corresponds to
    # the optimal w so we have w opt. = [-0.19,-0.02,-1.10,-0.80]
    # and the optimal b = 8.17
    
    Wopt= numpy.array(x[0:-1])
    bopt = x[-1]
    ret_args= [Wopt, bopt,prior]
    return ret_args
    # print(Wopt)
    # print(bopt)

#print(d)

#######################################
# all the previous is the training part 
#as we got the Wopt and bopt now we use them in
# classifying

def evaluate(DTE,r_args):
    Wopt = r_args[0]
    bopt = r_args[1]
    prior = r_args[2]
    #first we will calculate the score matrix
    #S =numpy.zeros(LTE.shape[0])
    predictedCls=numpy.zeros(DTE.shape[1],dtype=int)
    
    # SS= numpy.dot(Wopt,DTE)+bopt + numpy.log(prior / (1-prior))
    
    SS= numpy.dot(Wopt,DTE)+bopt
    
    for i in range(DTE.shape[1]):
        #S[i]= numpy.dot(Wopt,DTE[:,i])+bopt
        predictedCls[i]=1 if SS[i]>0 else 0
        
    return predictedCls ,SS


#print(S)
#print(SS)
# print(predictedCls)


############################################
# now we check our job
# the evaluation part comparing my prediction 
# with the class labels LTE

# tot_n = LTE.shape[0]
# wrong = 0

# for i in range(tot_n):
#     if predictedCls[i] != LTE[i]:
#         wrong+=1

# print("Error Rate is ")

# print(str(wrong/tot_n*100)+" %")



