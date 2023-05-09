import numpy
import scipy
import scipy.linalg
import math


def load(File):
    Data = []
    Labels = []
    with open(File) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1]
                Data.append(attrs)
                Labels.append(label)
            except:
                pass    
    Data = numpy.hstack(Data)
    Labels = numpy.array(Labels, dtype=numpy.int32)
    return Data, Labels


def mcol (lst):
    return lst.reshape((lst.shape[0],1))

def mrow (lst):
    return lst.reshape((1,lst.shape[0]))

def unique(list1): 
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)       
    return sorted(unique_list)

