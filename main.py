import matplotlib.pyplot as plt
import numpy 
from Library.lib import * 
import seaborn as sb
from scipy.stats import norm



########################## Loading Datasets ##########################################
DTR, LTR = load('data/Train.txt')
DTE, LTE = load('data/Test.txt')



def features_Analysis(DTR,LTR,DTE):
    L0_DTR = DTR[:,LTR==0] #Male class
    L1_DTR = DTR[:,LTR==1] #Female class
    
    print("The number of Training samples for the class Male are "+str(L0_DTR.shape[1]))
    print("The number of Training samples for the class Female are "+str(L1_DTR.shape[1]))
    
    print("The number of Features are "+str(DTR.shape[0]))
    
    for i in range(DTR.shape[0]):
        
        plt.figure('features Distribution')
        plt.xlabel("Feature"+str(i+1))
        plt.ylabel("Contibution")
        plt.hist(L0_DTR[i,:],bins = 40,density=True,label='Male',edgecolor='b',alpha=0.65)
        plt.hist(L1_DTR[i,:],bins = 40,density=True,label='Female',edgecolor='r',alpha=0.65)
        plt.legend(loc='upper right')
        plt.show()   
   
    # Gaussianize the Training features 
    y = numpy.zeros(DTR.shape)
    
    for i in range(DTR.shape[1]) :
        x = DTR[:,i]
        count = numpy.sum(x.reshape((x.shape[0],1))>DTR,axis=1)
        rank = (count+1)/ (DTR.shape[1]+2)
        y[:,i]= norm.ppf(rank)
     
    
    L0_ = y[:,LTR==0] #Male class
    L1_ = y[:,LTR==1] #Female class
    
    
    for i in range(DTR.shape[0]):
        
        plt.figure('features Distribution')
        plt.xlabel("Feature"+str(i+1))
        plt.ylabel("Contibution")
        plt.hist(L0_[i,:],bins = 40,density=True,label='Male',edgecolor='b',alpha=0.65)
        plt.hist(L1_[i,:],bins = 40,density=True,label='Female',edgecolor='r',alpha=0.65)
        plt.legend(loc='upper right')
        plt.show()   
   
    # Displaying heatmap
    allDataset = abs(numpy.corrcoef(DTR))
    sb.heatmap(allDataset, cmap="Greys", annot=True)
    plt.xlabel("Total Data Features Correlation")
    plt.ylabel("Raw Data")
    
    plt.show()
    posSamples = abs(numpy.corrcoef(L1_DTR))
    sb.heatmap(posSamples, cmap="Reds", annot=True)
    plt.xlabel("Female Class Freatures Correlation")
    plt.ylabel("Raw Data")
    plt.show()
    negSamples = abs(numpy.corrcoef(L0_DTR))
    sb.heatmap(negSamples, cmap="Blues", annot=True)
    plt.xlabel("Male Class Freatures Correlation")
    plt.ylabel("Raw Data")
    plt.show()
    
    # displaying heatmap after Gaussianization
   
    allDataset = abs(numpy.corrcoef(y))
    sb.heatmap(allDataset, cmap="Greys", annot=True)
    plt.xlabel("Total Data Freatures Correlation")
    plt.ylabel("Gaussianized Data")
    plt.show()
    posSamples = abs(numpy.corrcoef(L1_))
    sb.heatmap(posSamples, cmap="Reds", annot=True)
    plt.xlabel("Female Class Freatures Correlation")
    plt.ylabel("Gaussianized Data")
    plt.show()
    negSamples = abs(numpy.corrcoef(L0_))
    sb.heatmap(negSamples, cmap="Blues", annot=True)
    plt.xlabel("Male Class Freatures Correlation")
    plt.ylabel("Gaussianized Data")
    plt.show()
    
    DTR=y
    return DTR,DTE

         

#------------------- main----------------------------#

# Feature analysis and Gaussianizing the training 
DTR_,DTE_ = features_Analysis(DTR,LTR,DTE)


